# Packages from PM4Py
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluation
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

# Other
import pandas as pd
from typing import Union
from metrics.utils.output_range import *
from metrics.utils.quality_criteria import *


@classify(QualityCriteria.FITNESS)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def token_based_fitness(log: Union[EventLog, pd.DataFrame], net: PetriNet, im: Marking, fm: Marking):
    """Calculates the fitness of a process model using token-based replay.

    Args:
        log (Union[EventLog, pd.DataFrame]): The event log containing the
            observed behavior.
        net (PetriNet): The Petri net model to evaluate.
        im (Marking): The initial marking (initial state) of the Petri net.
        fm (Marking): The final marking (final state) of the Petri net.

    Returns:
         Dict[str, float]: The token-based fitness score, normalized between 0.0 and 1.0.

    References:
        - Rozinat, A., & van der Aalst, W. M. P. (2005). Conformance testing:
          Measuring the fit and appropriateness of event logs and process models.
    """
    from pm4py.algo.evaluation.replay_fitness.algorithm import Variants as FitnessVariants
    fit = fitness_evaluation.apply(
        log=log,
        petri_net=net,
        initial_marking=im,
        final_marking=fm,
        variant=FitnessVariants.TOKEN_BASED
    )
    return fit
#enddef

@classify(QualityCriteria.FITNESS)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def alignment_based_fitness(log: Union[EventLog, pd.DataFrame], net: PetriNet, im: Marking, fm: Marking):
    """Calculates the fitness of a process model using alignments.

    Args:
        log (Union[EventLog, pd.DataFrame]): The event log containing the
            observed behavior.
        net (PetriNet): The Petri net model to evaluate.
        im (Marking): The initial marking (initial state) of the Petri net.
        fm (Marking): The final marking (final state) of the Petri net.

    Returns:
        Dict[str, float]: The alignment-based fitness score, normalized between 0.0 and 1.0.

    References:
        - van der Aalst, W. M. P. (2012). Replaying History on Process Models
          for Conformance Checking and Performance Analysis.
    """
    from pm4py.algo.evaluation.replay_fitness.algorithm import Variants as FitnessVariants
    fit = fitness_evaluation.apply(
        log=log,
        petri_net=net,
        initial_marking=im,
        final_marking=fm,
        variant=FitnessVariants.ALIGNMENT_BASED
    )
    return fit
#enddef