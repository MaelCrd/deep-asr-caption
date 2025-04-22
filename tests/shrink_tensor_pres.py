# def shrink_list_preserve_order(original_list, new_size):
#     if new_size >= len(original_list):
#         return original_list  # No need to shrink if new size is larger or equal

#     # Calculate the step size to pick elements
#     step = len(original_list) / new_size

#     # Use a list to keep track of the indices to pick
#     indices_to_pick = [round(i * step) for i in range(new_size)]

#     # Create the shrunken list by picking elements at the calculated indices
#     shrunken_list = [original_list[i] for i in indices_to_pick]

#     return shrunken_list

# # Example usage
# original_list = [1, 1, 1, 1, 3, 3, 3, 3, 8, 8, 8, 1, 1, 1, 1, 1, 1]
# new_size = 10
# shrunken_list = shrink_list_preserve_order(original_list, new_size)
# print(shrunken_list)

import torch

def shrink_tensor_preserve_order(original_tensor, new_size):
    if new_size >= original_tensor.size(0):
        return original_tensor  # No need to shrink if new size is larger or equal

    # Calculate the step size to pick elements
    step = original_tensor.size(0) / new_size

    # Use a list to keep track of the indices to pick
    indices_to_pick = [round(i * step) for i in range(new_size)]

    # Create the shrunken tensor by picking elements at the calculated indices
    shrunken_tensor = original_tensor[indices_to_pick]

    return shrunken_tensor
    
    # return original_tensor[::original_tensor.size(0)//new_size]


if __name__ == "__main__":
    # Example usage
    original_tensor = torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 8, 8, 8, 1, 1, 1, 1, 1, 1])
    new_size = 100
    shrunken_tensor = shrink_tensor_preserve_order(original_tensor, new_size)
    print(shrunken_tensor)

