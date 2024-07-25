import torch
from .attention_aggregation_method import AttentionAggregationMethod


class AttentionMatrix:
    """
    Represents a self-attention matrix, recording the number of layers and heads,
    as well as whether it has been formatted for visualization or not.

    """

    def __init__(self, attention_matrix):
        """
        `AttentionMatrix` constructor.

        Args:
            attention_matrix: out-of-the-box attention matrix from the model's generation methods
                has shape: `num_response_tokens x num_layers x 1 x num_heads x a x b`

                `a x b` is either `num_prompt_tokens x num_prompt_tokens` (for the first response token only)
                    or `1 x num_prompt_tokens` otherwise self.num_layers = len(attention_matrix[0])
        """
        self.attention_matrix = attention_matrix  # out-of-the-box, has shape: `num_response_tokens x num_layers x 1 x num_heads x a x b` where `a x b` is either `num_prompt_tokens x num_prompt_tokens` (for the first response token only), or `1 x num_prompt_tokens` otherwise.
        self.num_layers = len(attention_matrix[0])
        self.num_heads = len(attention_matrix[0][0][0])
        self.is_formatted = False

    def format(
        self, aggr_method: AttentionAggregationMethod, zero_first_attention: bool
    ) -> None:
        """
        Formats the wrapped attention matrix for HTML visualization, aggregating it based on the specified aggregation method.

        Args:
            aggr_method: the aggregation method of the attention matrix. See `AttentionAggregationMethod`.

            zero_first_attention: whether to ignore self attention values towards the first token.
        """

        if self.is_formatted:
            pass

        self.is_formatted = True
        res = []

        ## attention has shape num_response_tokens x 1 x num_heads x a x b
        num_response_tokens = len(self.attention_matrix)

        for i in range(num_response_tokens):

            token_attention = self.attention_matrix[i]

            squeezed = []
            for layer_attention in token_attention:
                layer_attention = torch.squeeze(layer_attention)  # num_heads x seq_len

                if i == 0:  # num_heads x seq_len x seq_len -> num_heads x seq_len
                    layer_attention = layer_attention[:, -1]

                if zero_first_attention:
                    layer_attention[:, 0] = torch.zeros(len(layer_attention))

                if aggr_method == AttentionAggregationMethod.NONE:
                    squeezed.append(layer_attention.tolist())  # num_heads x seq_len
                elif aggr_method == AttentionAggregationMethod.HEADWISE_AVERAGING:
                    squeezed.append(
                        torch.mean(
                            layer_attention, 0, keepdim=True
                        ).tolist()  # 1 x seq_len
                    )  ## Keep only attentions related to the prompt tokens

            # squeezed has shape num_layers x {num_heads if aggr_method is NONE else 1 for HEADWISE_AVERAGING} x {num_tokens_before_current_one}
            res.append(squeezed)

        ## res has shape: num_res_tokens x num_layers x num_heads x {varying: num_tokens_before_current_one}
        nh = len(res[0][0])
        nl = len(res[0])
        tmp1 = [
            [[token_attention[l][h] for token_attention in res] for h in range(nh)]
            for l in range(nl)
        ]

        assert len(tmp1) == nl
        assert len(tmp1[0]) == nh
        assert len(tmp1[0][0]) == len(res)

        self.attention_matrix = tmp1
        self.num_heads = nh
        self.num_layers = nl

    def __repr__(self):
        """
        Debugging string representation of `AttentionMatrix`
        """
        return f"AttentionMatrix ({self.num_heads} head(s), {self.num_layers} layer(s))"

    def __str__(self):
        """
        Regular string representation of `AttentionMatrix`
        """
        return self.__repr__()

    def __eq__(self, other):
        """Overrides equality comparison with another attention matrix"""
        if not isinstance(other, AttentionMatrix):
            return False

        return (
            self.num_heads == other.num_heads
            and self.num_layers == other.num_layers
            and self.is_formatted == other.is_formatted
            and len(self.attention_matrix) == len(other.attention_matrix)
        )  ## TODO find a way to compare content as well
