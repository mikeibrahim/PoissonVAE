from base.utils_model import *
from scipy import stats as sp_stats
dists.Distribution.set_default_validate_args(False)


# ============================================================================
# Indicator Functions for Poisson rsample
# ============================================================================
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
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
	return 3 * u.pow(2) - 2 * u.pow(3)


def cosine_sigmoid(x: torch.Tensor) -> torch.Tensor:
	"""
	Cosine-based smooth approximation.
	Maps [-1, 1] to [0, 1] with C_infinity smoothness inside the window.
	f(u) = 0.5 * (1 - cos(pi * u)), where u = (x+1)/2
	"""
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
	return 0.5 * (1.0 - torch.cos(torch.pi * u))


_INDICATOR_FNS = {
	'sigmoid': torch.sigmoid,
	'linear': hard_sigmoid,
	'cubic': cubic_sigmoid,
	'cosine': cosine_sigmoid,
}


def compute_n_exp(rate: float, p: float = 1e-6):
	"""Compute n_exp from rate using Poisson quantile."""
	assert rate > 0.0, f"must be positive, got: {rate}"
	pois = sp_stats.poisson(rate)
	n_exp = pois.ppf(1.0 - p)
	return int(n_exp)


# noinspection PyAbstractClass
class Categorical(dists.RelaxedOneHotCategorical):
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


# noinspection PyAbstractClass
class Laplace(dists.Laplace):
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
class Normal(dists.Normal):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			seed: int = None,
			device: torch.device = None,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(log_scale, clamp)
		super(Normal, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.t = temp
		self.c = clamp
		self._init_rng(seed, device)

	def kl(self, p: dists.Normal = None):
		if p is None:
			term1 = self.mean
			term2 = self.scale
		else:
			term1 = (self.mean - p.mean) / p.scale
			term2 = self.scale / p.scale
		kl = 0.5 * (
			term1.pow(2) + term2.pow(2) +
			torch.log(term2).mul(-2) - 1
		)
		return kl

	@torch.inference_mode()
	def sample(self, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		samples = torch.normal(
			mean=self.loc.expand(shape),
			std=self.scale.expand(shape),
			generator=self.rng,
		)
		return samples

	def _init_rng(self, seed, device):
		if seed is not None:
			self.rng = torch.Generator(device)
			self.rng.manual_seed(seed)
		else:
			self.rng = None
		return


# noinspection PyTypeChecker
class Poisson:
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 1.0,
			n_exp: int | str = 263,
			clamp: float = 5.3,
			indicator_approx: str = 'sigmoid',
			n_exp_p: float = 1e-3,
	):
		assert temp >= 0
		assert indicator_approx in _INDICATOR_FNS
		self.t = temp
		self.c = clamp
		self.indicator_approx = indicator_approx
		self.n_exp_p = n_exp_p
		self._init(log_rate)
		# compute n_exp after init (need rate)
		if n_exp == 'infer':
			n_exp = self._infer_n_exp(n_exp_p)
		self.n = int(n_exp)

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

	def rsample(self, hard: bool = False):
		x = self.exp.rsample((self.n,))  # inter-event times
		times = torch.cumsum(x, dim=0)   # arrival times of events

		indicator = times < 1.0
		z_hard = indicator.sum(0).float()

		if self.t > 0:
			# compute raw logits for sigmoid-like function
			logits = (1.0 - times) / self.t
			fn = _INDICATOR_FNS.get(self.indicator_approx)
			indicator = fn(logits)
			z = indicator.sum(0).float()
		else:
			z = z_hard

		if hard:
			return z + (z_hard - z).detach()
		return z

	def sample(self):
		return torch.poisson(self.rate).float()

	def log_p(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)

	def _init(self, log_rates):
		eps = torch.finfo(torch.float32).eps
		log_rates = softclamp_upper(log_rates, self.c)
		self.rate = torch.exp(log_rates) + eps
		self.exp = dists.Exponential(self.rate)
		return


class GumbelSoftmaxPoisson:
	"""
	Gumbel-Softmax relaxation for Poisson distribution.
	Uses categorical approximation with softmax over count values.
	"""
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			upperbound_method: str = "fixed",
			upperbound_param: int | float = 5,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		self.t = temp
		self.c = clamp
		self.upperbound_method = upperbound_method
		self.upperbound_param = upperbound_param
		self._init(log_rate)
		self._compute_upperbound()

	def _init(self, log_rates):
		eps = torch.finfo(torch.float32).eps
		if self.c is not None:
			log_rates = softclamp_upper(log_rates, self.c)
		self.log_rate = log_rates
		self.rate = torch.exp(log_rates) + eps
		return

	def _compute_upperbound(self):
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
		else:
			raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")

	def __repr__(self):
		parts = [
			f"rate: {self.rate.shape}",
			f"temp: {self.t}",
			f"upperbound: {self.upperbound}",
		]
		return f"GumbelSoftmaxPoisson({', '.join(parts)})"

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

	def rsample(self, hard: bool = False, n_samples: int | None = None):
		if self.t == 0.0:
			return self.sample(n_samples=n_samples)
		logit_pi = self.logit_pi
		if n_samples is not None:
			logit_pi = logit_pi.unsqueeze(0).expand(n_samples, -1, -1)
		z = F.gumbel_softmax(
			logits=logit_pi,
			tau=self.t,
			hard=hard,
		)  # (..., upperbound)
		# Convert to expected count
		return self.aggregate_samples(z)

	def aggregate_samples(self, gumbel_samples: torch.Tensor):
		k = torch.arange(self.upperbound, device=self.rate.device).float()
		return gumbel_samples @ k

	def log_p(self, samples: torch.Tensor, eps: float = 1e-8):
		"""Log probability (approximation for Gumbel-Softmax samples)."""
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)


def softclamp_sym(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


def softclamp_upper(x: torch.Tensor, c: float):
	return c - F.softplus(c - x)


def softclamp(x: torch.Tensor, upper: float, lower: float = 0.0):
	return lower + F.softplus(x - lower) - F.softplus(x - upper)
