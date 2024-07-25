from enum import Enum


class AttentionAggregationMethod(Enum):
    """
    Represents the possible attention aggregation methods.
    The supported aggregation methods are:
        - `NONE`: no dimension collapses
        - `HEADWISE_AVERAGING`: head dimension is collapsed
    """

    NONE = 1
    """ Represents the empty aggregation method - all layer and head dimensions of the attention matrix are kept. """

    HEADWISE_AVERAGING = 2
    """ Represents the headwise averaging aggregation method - the head dimension of the attention matrix collapses, while the layer dimension is kept. """
