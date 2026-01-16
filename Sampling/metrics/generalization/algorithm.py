from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluation
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Union
import pandas as pd
from metrics.utils.output_range import *
from metrics.utils.quality_criteria import *


@classify(QualityCriteria.GENERALIZATION)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def generalization(log: Union[EventLog, EventStream, pd.DataFrame], net: PetriNet, im: Marking, fm: Marking) -> float:
    """Calculates the generalization of a process model.

    Args:
        log (Union[EventLog, EventStream, pd.DataFrame]): The event log
            containing the observed behavior.
        net (PetriNet): The Petri net model to evaluate.
        im (Marking): The initial marking (initial state) of the Petri net.
        fm (Marking): The final marking (final state) of the Petri net.

    Returns:
        float: The generalization score, typically normalized between 0.0 and 1.0.

    References:
        - Buijs, J. C. A. M., et al. (2014). Quality dimensions in process
          discovery: The importance of fitness, precision, generalization
          and simplicity.
    """
    from pm4py.algo.evaluation.generalization.variants import token_based
    g = generalization_evaluation.apply(log=log, petri_net=net, initial_marking=im, final_marking=fm, variant=token_based)
    return g
#enddef