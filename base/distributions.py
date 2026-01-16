from base.utils_model import *
dists.Distribution.set_default_validate_args(False)


class Poisson:
	"""
	Poisson distribution with differentiable (soft) sampling via the Exponential trick.
	
	This implementation uses the inverse CDF method for Poisson sampling:
	1. Sample inter-arrival times from Exp(rate) distribution
	2. Compute cumulative arrival times
	3. Count events arriving before time t=1 using a soft indicator function
	
	The soft indicator allows gradient flow during training when temp > 0.
	At temp=0 (or hard=True), returns exact Poisson samples.
	
	Attributes:
		rate (Tensor): Poisson rate parameter λ (mean of the distribution)
		temp (float): Temperature for soft counting. 0 = hard sampling
		n_exp (int): Number of exponential samples for soft counting (truncation level)
		indicator_approx (str): Type of soft indicator function ('sigmoid', 'linear', 'cubic', 'cosine')
	
	Example:
		>>> dist = Poisson(log_rate=torch.zeros(100), temp=0.1)
		>>> samples = dist.rsample()  # soft samples for training
		>>> samples = dist.sample()   # hard samples for evaluation
	"""
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 0.0,
			clamp: float | None = None,
			indicator_approx: str = 'sigmoid',
			n_exp: int | str = 'infer',
			n_exp_p: float = 1e-3,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		assert indicator_approx in _INDICATOR_FNS
		self.indicator_approx = indicator_approx
		self.temp = float(temp)
		self.clamp = float(clamp) if clamp is not None else None
		# setup rate & exp dist
		if clamp is not None:
			log_rate = softclamp_upper(
				log_rate, clamp)
		eps = torch.finfo(torch.float32).eps
		self.rate = torch.exp(log_rate) + eps
		self._exp = dists.Exponential(self.rate)
		# compute n_exp
		if n_exp == 'infer':
			n_exp = self._infer_n_exp(n_exp_p)
		self.n_exp = int(n_exp)

	def __repr__(self):
		parts = [
			f"rate: {self.rate.shape}",
			f"temp: {self.temp}",
			f"n_exp: {self.n_exp}",
		]
		return f"Poisson({', '.join(parts)})"

	@torch.no_grad()
	def _infer_n_exp(self, n_exp_p):
		max_rate = self.rate.max().item()
		n_exp = compute_n_exp(max_rate, n_exp_p)
		return int(n_exp)

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	# noinspection PyTypeChecker
	def rsample(self, hard: bool = False):
		if self.temp == 0.0 or hard:
			return self.sample()

		# (1) inter-event times
		x = self._exp.rsample((self.n_exp,))

		# (2) arrival t of events
		times = torch.cumsum(x, dim=0)

		# (3) compute raw logits
		# (input to the sigmoid-like function)
		# This maps the threshold time t=1 to 0
		# t = 1 - temp → logits = 1
		# t = 1 + temp → logits = -1
		logits = (1 - times) / self.temp

		# (4) events within [0, 1]
		fn = _INDICATOR_FNS.get(
			self.indicator_approx)
		indicator = fn(logits)

		# (5) soft event counts
		z = indicator.sum(0).float()

		return z

	@torch.no_grad()
	def sample(self):
		return torch.poisson(self.rate).float()

	def log_prob(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)


class GumbelSoftmaxPoisson:
	"""
	Poisson distribution with differentiable sampling via Gumbel-Softmax relaxation.
	
	Instead of sampling discrete counts directly, this treats the Poisson as a
	categorical distribution over {0, 1, ..., upperbound-1} and uses Gumbel-Softmax
	to obtain differentiable samples.
	
	The upperbound can be determined by several methods:
	- 'fixed': Use a constant upperbound value
	- 'std_ratio': upperbound = max(sqrt(rate)) * param (scales with rate variance)
	- 'quantile': upperbound = Poisson quantile at probability 1-param
	- 'adaptive': Dynamically updated during training based on running average of max rates
	
	For 'adaptive' method, the model maintains a running estimate of the max rate seen
	during training, providing robustness to outliers while adapting to the actual
	rate distribution. The upperbound is computed as the Poisson quantile at p=upperbound_param.
	
	Attributes:
		log_rate (Tensor): Log of the Poisson rate parameter
		temp (float): Temperature for Gumbel-Softmax. Lower = closer to discrete
		upperbound_method (str): Method for computing upperbound ('fixed', 'std_ratio', 'quantile', 'adaptive')
		upperbound_param (int|float): Parameter for the upperbound method
		upperbound (int): Computed upperbound value (max count + 1)
	
	Example:
		>>> dist = GumbelSoftmaxPoisson(log_rate=torch.zeros(100), temp=0.5, upperbound_method='fixed', upperbound_param=50)
		>>> soft_samples = dist.rsample()  # returns (..., upperbound) Gumbel-Softmax samples
		>>> counts = dist.aggregate_samples(soft_samples)  # convert to expected counts
	"""
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 1.0,
			upperbound_method: str = "fixed",
			upperbound_param: int | float = 5,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		self.temp = temp
		self.log_rate = log_rate
		self.upperbound_method = upperbound_method
		self.upperbound_param = upperbound_param
		if self.upperbound_method == "fixed":
			self.upperbound = int(self.upperbound_param)
		elif self.upperbound_method == "std_ratio":
			self.upperbound = int(
				self.rate.detach().sqrt().cpu().numpy().max() * self.upperbound_param
			)
		elif self.upperbound_method == "quantile":
			self.upperbound = compute_n_exp(
				rate=self.rate.detach().cpu().numpy().max(),
				p=self.upperbound_param,
			)
		elif self.upperbound_method == "adaptive":
			# For adaptive mode, upperbound_param is the quantile probability (e.g., 1e-3)
			# The actual upperbound is set externally by the model based on running average
			# Default to a reasonable value; will be updated during training
			self.upperbound = int(self.upperbound_param) if isinstance(self.upperbound_param, int) else 50
		else:
			raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")

	def __repr__(self):
		parts = [
			f"rate: {self.rate.shape}",
			f"temp: {self.temp}",
			f"upperbound: {self.upperbound}",
		]
		return f"Poisson({', '.join(parts)})"

	@property
	def rate(self):
		return torch.exp(self.log_rate)

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	@property
	def logit_pi(self):
		k = torch.arange(self.upperbound, device=self.rate.device)
		return k * self.log_rate.unsqueeze(-1) - torch.lgamma(k + 1)

	@torch.no_grad()
	def sample(self, n_samples: int | None = None):
		if n_samples is not None:
			rate = self.rate.unsqueeze(0).expand(n_samples, -1)
			return torch.poisson(rate).float()
		return torch.poisson(self.rate).float()

	def rsample(self, n_samples: int | None = None):
		if self.temp == 0.0:
			return self.sample(n_samples=n_samples)
		logit_pi = self.logit_pi
		if n_samples is not None:
			logit_pi = logit_pi.unsqueeze(0).expand(n_samples, -1, -1)
		z = F.gumbel_softmax(
			logits=logit_pi,
			tau=self.temp,
			hard=False,
		)  # (..., upperbound)
		return z

	def aggregate_samples(self, gumbel_samples: torch.Tensor):
		k = torch.arange(self.upperbound, device=self.rate.device).float()
		return gumbel_samples @ k

	def log_prob(self, samples: torch.Tensor, eps: float = 1e-8):
		ln_x = torch.clamp(samples, min=eps).log()
		logit_pi = self.logit_pi
		return (
				torch.lgamma(torch.tensor(self.upperbound))
				+ (self.upperbound - 1) * torch.tensor(self.temp).log()
				- self.upperbound * torch.logsumexp(logit_pi - self.temp * ln_x, dim=-1)
				+ (logit_pi - (self.temp + 1) * ln_x).sum(dim=-1)
		)


# Used to construct decoder
# noinspection PyAbstractClass
class Normal(dists.Normal):
	"""
	Gaussian/Normal distribution with optional temperature scaling and log-scale clamping.
	
	Extends PyTorch's Normal distribution with:
	- Soft clamping on log_scale to prevent numerical instability
	- Temperature scaling of the standard deviation
	- KL divergence computation against a prior
	- Noise retrieval for reconstructing the reparameterization noise
	
	The temperature parameter scales the variance: scale_effective = scale * temp.
	This is useful for controlling the stochasticity during training vs evaluation.
	
	Attributes:
		loc (Tensor): Mean of the distribution
		scale (Tensor): Standard deviation (after clamping and temp scaling)
		temp (float): Temperature multiplier for scale
		clamp (float): Soft clamping value for log_scale (None = no clamping)
	
	Example:
		>>> dist = Normal(loc=mu, log_scale=log_sigma, temp=1.0, clamp=5.0)
		>>> samples = dist.rsample()
		>>> kl = dist.kl()  # KL divergence against standard normal
	"""
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float | None = None,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(x=log_scale, c=clamp)
		scale = torch.exp(log_scale)

		if temp != 1.0:
			scale = scale * temp

		super().__init__(loc=loc, scale=scale, **kwargs)

		self.temp = temp
		self.clamp = clamp

	def kl(self, p: dists.Normal = None):
		if p is None:
			term1 = self.mean
			term2 = self.scale
		else:
			term1 = (self.mean - p.mean) / p.scale
			term2 = self.scale / p.scale
		kl = 0.5 * (term1.pow(2) + term2.pow(2) + torch.log(term2).mul(-2) - 1)
		return kl

	@torch.no_grad()
	def retrieve_noise(self, samples: torch.Tensor):
		return (samples - self.loc).div(self.scale)


# noinspection PyAbstractClass
class Laplace(dists.Laplace):
	"""
	Laplace distribution with optional temperature scaling and log-scale clamping.
	
	Extends PyTorch's Laplace distribution with:
	- Soft symmetric clamping on log_scale to prevent numerical instability
	- Temperature scaling of the scale parameter
	- KL divergence computation against a prior
	
	The Laplace distribution has heavier tails than Gaussian, making it useful
	for sparse representations where many values are near zero but some are large.
	
	Attributes:
		loc (Tensor): Mean/mode of the distribution
		scale (Tensor): Scale parameter (after clamping and temp scaling)
		t (float): Temperature multiplier for scale
		c (float): Clamping value for log_scale
	
	Example:
		>>> dist = Laplace(loc=mu, log_scale=log_b, temp=1.0, clamp=5.3)
		>>> samples = dist.rsample()
		>>> kl = dist.kl()  # KL divergence against standard Laplace
	"""
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(log_scale, clamp)
		super(Laplace, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.t = temp
		self.c = clamp

	def kl(self, p: dists.Laplace = None):
		if p is not None:
			mean, scale = p.mean, p.scale
		else:
			mean, scale = 0, 1

		delta_m = torch.abs(self.mean - mean)
		delta_b = self.scale / scale
		term1 = delta_m / self.scale
		term2 = delta_m / scale

		kl = (
			delta_b * torch.exp(-term1) +
			term2 - torch.log(delta_b) - 1
		)
		return kl


# noinspection PyAbstractClass
class Categorical(dists.RelaxedOneHotCategorical):
	"""
	Categorical distribution with Gumbel-Softmax relaxation for differentiable sampling.
	
	Extends PyTorch's RelaxedOneHotCategorical to provide:
	- Temperature-controlled relaxation (lower temp = closer to discrete)
	- Mean and variance properties for the probability simplex
	- KL divergence computation against a uniform prior
	
	Used in categorical VAEs where latent variables are discrete categories.
	The Gumbel-Softmax trick allows gradients to flow through discrete choices.
	
	Attributes:
		logits (Tensor): Unnormalized log-probabilities for each category
		temperature (float): Gumbel-Softmax temperature (clamped to eps minimum)
		probs (Tensor): Normalized probabilities (softmax of logits)
	
	Example:
		>>> dist = Categorical(logits=torch.randn(batch, n_categories), temp=0.5)
		>>> soft_samples = dist.rsample()  # differentiable soft one-hot
		>>> kl = dist.kl()  # KL against uniform distribution
	"""
	def __init__(
			self,
			logits: torch.Tensor,
			temp: float = 1.0,
			**kwargs,
	):
		self._categorical = None
		temp = max(temp, torch.finfo(torch.float).eps)
		super(Categorical, self).__init__(
			logits=logits, temperature=temp, **kwargs)

	@property
	def t(self):
		return self.temperature

	@property
	def mean(self):
		return self.probs

	@property
	def variance(self):
		return self.probs * (1 - self.probs)

	def kl(self, p: dists.Categorical = None):
		if p is None:
			probs = torch.full(
				size=self.probs.size(),
				fill_value=1 / self.probs.size(-1),
			)
			p = dists.Categorical(probs=probs)
		q = dists.Categorical(probs=self.probs)
		return dists.kl.kl_divergence(q, p)


def compute_n_exp(rate: float, p: float = 1e-6):
	"""
	Compute the truncation level for Poisson sampling.
	
	Returns the smallest n such that P(X > n) < p for X ~ Poisson(rate).
	This is the (1-p) quantile of the Poisson distribution.
	
	Used to determine how many exponential samples are needed to accurately
	approximate Poisson sampling via the inverse CDF method.
	
	Args:
		rate: Poisson rate parameter (must be positive)
		p: Tail probability threshold (default: 1e-6)
	
	Returns:
		int: Truncation level n_exp
	
	Example:
		>>> n = compute_n_exp(rate=10.0, p=1e-6)  # ~25 for rate=10
	"""
	assert rate > 0.0, f"must be positive, got: {rate}"
	pois = sp_stats.poisson(rate)
	n_exp = pois.ppf(1.0 - p)
	return int(n_exp)


def softclamp_upper(x: torch.Tensor, c: float):
	return c - F.softplus(c - x)


def softclamp_sym(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


def softclamp(x: torch.Tensor, upper: float, lower: float = 0.0):
	return lower + F.softplus(x - lower) - F.softplus(x - upper)


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
	"""
	Piecewise linear approximation (Hard Sigmoid).
	Maps [-1, 1] linearly to [0, 1].
	Exact 0 for x < -1, Exact 1 for x > 1.
	"""
	return torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)


def cubic_sigmoid(x: torch.Tensor) -> torch.Tensor:
	"""
	Cubic Hermite interpolation (Smoothstep).
	Maps [-1, 1] to [0, 1] with C1 smoothness (zero derivative at boundaries).
	f(u) = 3u^2 - 2u^3, where u = (x+1)/2
	"""
	# 1. Normalize x from [-1, 1] to u in [0, 1]
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
	# 2. Cubic polynomial
	return 3 * u.pow(2) - 2 * u.pow(3)


def cosine_sigmoid(x: torch.Tensor) -> torch.Tensor:
	"""
	Cosine-based smooth approximation.
	Maps [-1, 1] to [0, 1] with C_infinity smoothness inside the window.
	f(u) = 0.5 * (1 - cos(pi * u)), where u = (x+1)/2
	"""
	# 1. Normalize x from [-1, 1] to u in [0, 1]
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
	# 2. Cosine ease-in-out
	return 0.5 * (1.0 - torch.cos(torch.pi * u))


_INDICATOR_FNS = {
    'sigmoid': torch.sigmoid,
    'linear': hard_sigmoid,
    'cubic': cubic_sigmoid,
    'cosine': cosine_sigmoid,
}
