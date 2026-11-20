import random
import numpy as np

from fairseq_signals.dataclass import ChoiceEnum

PERTURBATION_CHOICES = ChoiceEnum(
    [
        "3kg",
        "random_leads_masking",
        "powerline_noise",
        "emg_noise",
        "baseline_shift",
        "baseline_wander",
    ]
)
MASKING_LEADS_STRATEGY_CHOICES = ChoiceEnum(["random", "conditional"])

def instantiate_from_name(str: PERTURBATION_CHOICES, **kwargs):
    if str == "random_leads_masking":
        return RandomLeadsMask(**kwargs)
    elif str == "powerline_noise":
        return PowerlineNoise(**kwargs)
    elif str == "emg_noise":
        return EMGNoise(**kwargs)
    elif str == "baseline_shift":
        return BaselineShift(**kwargs)
    elif str == "baseline_wander":
        return BaselineWander(**kwargs)
    else:
        raise ValueError(f"inappropriate perturbation choices: {str}")

def adjust_channel_dependency(ecg):
    # synthesize III, aVR, aVL, aVF from I, II
    if ecg.shape[0] != 12:
        raise ValueError(
            "task.dependency=true synthesizes III, aVR, aVL, and aVF from I and II, so it requires "
            "12-lead samples in the standard order (I, II, III, aVR, aVL, aVF, V1-V6), but got a "
            f"sample with {ecg.shape[0]} leads. Set task.dependency=false if only some leads are "
            "loaded (e.g., with task.leads_to_load)."
        )
    ecg[2] = ecg[1]-ecg[0]
    ecg[3] = -(ecg[1]+ecg[0])/2
    ecg[4] = ecg[0]-ecg[1]/2
    ecg[5] = ecg[1]-ecg[0]/2
    return ecg

class PowerlineNoise(object):
    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0,
        p=1.0,
        freq=500,
        dependency=True,
        **kwargs,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.freq = freq
        self.p = p
        self.dependency = dependency

    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0, 1):
            csz, tsz = new_sample.shape
            amp = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(1,1))
            f = 50 if np.random.uniform(0,1) > 0.5 else 60
            noise = self._apply_powerline_noise(tsz, f)
            new_sample = new_sample + noise * amp
            if self.dependency:
                new_sample = adjust_channel_dependency(new_sample)
        return new_sample.float()

    def _apply_powerline_noise(self, tsz, f):
        t = np.linspace(0,tsz-1,tsz)
        phase = np.random.uniform(0,2*np.pi)
        noise = np.cos(2*np.pi*f*(t/self.freq)+phase)
        return noise

class EMGNoise(object):
    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0,
        dependency=True,
        p=1.0,
        **kwargs,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.p = p
        self.dependency = dependency
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            csz, tsz = new_sample.shape
            amp = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(csz, 1))
            noise = np.random.normal(0, 1, [csz, tsz])
            new_sample = new_sample + noise * amp
            if self.dependency:
                new_sample = adjust_channel_dependency(new_sample)
        return new_sample.float()

class BaselineShift(object):
    def __init__(
        self,
        max_amplitude=0.25,
        min_amplitude=0,
        shift_ratio=0.2,
        num_segment=1,
        freq=500,
        dependency=False,
        p=1.0,
        **kwargs,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.shift_ratio = shift_ratio
        self.num_segment = num_segment
        self.freq = freq
        self.p = p
        self.dependency = dependency
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            csz, tsz = new_sample.shape
            shift_length = tsz * self.shift_ratio
            amp_channel = np.random.choice([1, -1], size=(csz, 1))
            amp_general = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(1,1))
            amp = amp_channel - amp_general
            noise = np.zeros(shape=(csz, tsz))
            for i in range(self.num_segment):
                segment_len = np.random.normal(shift_length, shift_length*0.2)
                t0 = int(np.random.uniform(0, tsz-segment_len))
                t = int(t0+segment_len)
                noise[:, t0:t] = 1
            new_sample = new_sample + noise * amp
            if self.dependency:
                new_sample = adjust_channel_dependency(new_sample)
        return new_sample.float()

class BaselineWander(object):
    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0,
        p=1.0,
        max_freq=0.2,
        min_freq=0.01,
        k=3,
        freq=500,
        dependency=True,
        **kwargs,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.k = k
        self.freq = freq
        self.p = p
        self.dependency = dependency
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            csz, tsz = new_sample.shape
            amp_channel = np.random.normal(1, 0.5, size=(csz, 1))
            amp_general = np.random.uniform(self.min_amplitude, self.max_amplitude, size=self.k)
            noise = np.zeros(shape=(1, tsz))
            for k in range(self.k):
                noise += self._apply_baseline_wander(tsz) * amp_general[k]
            noise = (noise * amp_channel).astype(np.float32)
            new_sample = new_sample + noise
            if self.dependency:
                new_sample = adjust_channel_dependency(new_sample)
        return new_sample.float()
        
    def _apply_baseline_wander(self, tsz):
        f = np.random.uniform(self.min_freq, self.max_freq)
        t = np.linspace(0, tsz-1, tsz)
        r = np.random.uniform(0, 2*np.pi)
        noise = np.cos(2*np.pi*f*(t/self.freq)+r)
        return noise

class RandomLeadsMask(object):
    def __init__(
        self,
        p=1,
        mask_leads_selection="random",
        mask_leads_prob=0.5,
        mask_leads_condition=None,
        **kwargs,
    ):
        if mask_leads_selection not in ("random", "conditional"):
            raise ValueError(
                "mask_leads_selection should be either 'random' or 'conditional', but got "
                f"{mask_leads_selection!r}."
            )
        if mask_leads_selection == "conditional":
            if mask_leads_condition is None or len(mask_leads_condition) != 2:
                raise ValueError(
                    "mask_leads_condition should be a tuple of 2 values for the conditional "
                    f"leads masking, but got {mask_leads_condition!r}."
                )
            n1, n2 = mask_leads_condition
            if not (0 <= n1 <= 6 and 0 <= n2 <= 6):
                raise ValueError(
                    "each value of mask_leads_condition should be between 0 and 6, but got "
                    f"({n1}, {n2})."
                )

        self.p = p
        self.mask_leads_prob = mask_leads_prob
        self.mask_leads_selection = mask_leads_selection
        self.mask_leads_condition = mask_leads_condition
    
    def __call__(self, sample):
        if self.p >= np.random.uniform(0,1):
            num_leads = sample.size(0)
            if self.mask_leads_selection == "random":
                survivors = np.random.uniform(0, 1, size=num_leads) >= self.mask_leads_prob
            else:
                # mask n1 leads out of the limb leads (I, II, III, aVR, aVL, aVF) and n2 leads out of
                # the precordial leads (V1-V6)
                if num_leads != 12:
                    raise ValueError(
                        "mask_leads_selection='conditional' requires 12-lead samples, but got a "
                        f"sample with {num_leads} leads. If only some leads are loaded with "
                        "task.leads_to_load, set task.enable_padding_leads=true to pad the other "
                        "leads, or use mask_leads_selection='random'."
                    )
                n1, n2 = self.mask_leads_condition
                survivors = np.zeros(num_leads, dtype=bool)
                survivors[random.sample(range(6), 6 - n1)] = True
                survivors[[6 + i for i in random.sample(range(6), 6 - n2)]] = True
            new_sample = sample.new_zeros(sample.size())
            new_sample[survivors] = sample[survivors]
        else:
            new_sample = sample.clone()
        return new_sample.float()