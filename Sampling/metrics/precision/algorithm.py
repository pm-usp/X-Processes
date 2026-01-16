from pm4py.algo.evaluation.precision import algorithm as precision_evaluation
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from metrics.utils.output_range import *
from metrics.utils.quality_criteria import *


@classify(QualityCriteria.PRECISION)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def etc_precision(log: EventLog, net: PetriNet, im: Marking, fm: Marking) -> float:
    """Calculates precision based on the ETConformance (ETC) algorithm.

    Args:
        log (EventLog): The event log containing the observed behavior.
        net (PetriNet): The Petri net model to evaluate.
        im (Marking): The initial marking (initial state) of the Petri net.
        fm (Marking): The final marking (final state) of the Petri net.

    Returns:
        float: The ETC precision score, normalized between 0.0 and 1.0.

    References:
        - Muñoz-Gama, J., & Carmona, J. (2010). A fresh look at precision in
          process conformance.
    """
    from pm4py.algo.evaluation.precision.variants import etconformance_token
    p = precision_evaluation.apply(log=log, net=net, marking=im, final_marking=fm,variant=etconformance_token)
    return p
#enddef

@classify(QualityCriteria.PRECISION)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def align_etc_precision(log: EventLog, net: PetriNet, im: Marking, fm: Marking) -> float:
    """Calculates precision using an alignments-based ETConformance approach.

    Args:
        log (EventLog): The event log containing the observed behavior.
        net (PetriNet): The Petri net model to evaluate.
        im (Marking): The initial marking (initial state) of the Petri net.
        fm (Marking): The final marking (final state) of the Petri net.

    Returns:
        float: The alignment-based ETC precision score, normalized between
        0.0 and 1.0.

    References:
        - Adriansyah, A. (2014). Measuring precision of modeled behavior.
        - Muñoz-Gama, J., & Carmona, J. (2010). A fresh look at precision in
          process conformance.
    """
    from pm4py.algo.evaluation.precision.variants import align_etconformance
    p = precision_evaluation.apply(log=log, net=net, marking=im, final_marking=fm,variant=align_etconformance)
    return p
#enddef