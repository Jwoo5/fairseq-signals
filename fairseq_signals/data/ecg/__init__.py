from fairseq_signals.dataclass import ChoiceEnum

PERTURBATION_CHOICES = ChoiceEnum(["3kg", "random_leads_masking", "none"])
MASKING_LEADS_STRATEGY_CHOICES = ChoiceEnum(["random", "conditional"])