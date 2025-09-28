import torch
from losses.utils import flatten


def test_flatten_basic_case():
    # input: batch_size=1, num_classes=2, height=2, width=2
    input_tensor = torch.tensor(
        [[
            [[1.0, 2.0], [3.0, 4.0]],   # class 0 scores
            [[5.0, 6.0], [7.0, 8.0]]    # class 1 scores
        ]]
    )  # shape [1, 2, 2, 2]

    target_tensor = torch.tensor(
        [[
            [0, 1],
            [1, -1]
        ]]
    )  # shape [1, 2, 2], with -1 as ignore index

    ignore_index = -1

    input_flatten, target_flatten = flatten(input_tensor,
                                            target_tensor, ignore_index)

    # After flatten:
    # positions: (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=ignore
    # => target_flatten should be [0,1,1]
    expected_targets = torch.tensor([0, 1, 1])

    assert torch.equal(target_flatten, expected_targets)

    # input_flatten should have same number of rows as targets, and num_cls=2
    assert input_flatten.shape == (3, 2)


def test_flatten_all_ignore():
    input_tensor = torch.randn(1, 3, 2, 2)
    target_tensor = torch.full((1, 2, 2), -1)  # everything is ignore
    ignore_index = -1

    input_flatten, target_flatten = flatten(input_tensor,
                                            target_tensor, ignore_index)

    # everything is filtered out
    assert input_flatten.numel() == 0
    assert target_flatten.numel() == 0
