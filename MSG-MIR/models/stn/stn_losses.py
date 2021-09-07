import torch
import torch.nn.functional as F


def smoothness_loss(deformation, img=None, alpha=0.0):
    """Calculate the smoothness loss of the given defromation field

    :param deformation: the input deformation
    :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    :param alpha: the alpha coefficient used in the bilateral filtering.
    :return:
    """
    # diff_1 = torch.abs(deformation[:, :, 2::, :] + deformation[:, :, 0:-2, :] - 2*deformation[:, :, 1:-1, :])
    # diff_2 = torch.abs((deformation[:, :, :, 2::] + deformation[:, :, :, 0:-2] - 2*deformation[:, :, :, 1:-1]))
    # diff_3 = torch.abs(deformation[:, :, 0:-2, 0:-2] + deformation[:, :, 2::, 2::] - 2*deformation[:, :, 1:-1, 1:-1])
    # diff_4 = torch.abs(deformation[:, :, 0:-2, 2::] + deformation[:, :, 2::, 0:-2] - 2*deformation[:, :, 1:-1, 1:-1])

    diff_1 = torch.abs(deformation[:, :, 1::, :] - deformation[:, :, 0:-1, :])
    diff_2 = torch.abs((deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1]))
    diff_3 = torch.abs(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1::, 1::])
    diff_4 = torch.abs(deformation[:, :, 0:-1, 1::] - deformation[:, :, 1::, 0:-1])
    if img is not None and alpha > 0.0:
        mask = img
        weight_1 = torch.exp(- alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
        weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
        weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
        weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
        weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        # weight_1 = torch.exp(- alpha * torch.abs(mask[:, :, 2::, :] + mask[:, :, 0:-2, :] - 2*mask[:, :, 1:-1, :]))
        # weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        # weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 2::] + mask[:, :, :, 0:-2] - 2*mask[:, :, :, 1:-1]))
        # weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        # weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-2, 0:-2] + mask[:, :, 2::, 2::] - 2*mask[:, :, 1:-1, 1:-1]))
        # weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        # weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-2, 2::] - mask[:, :, 2::, 0:-2] - 2*mask[:, :, 1::-1, 1:-1]))
        # weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    else:
        weight_1 = weight_2 = weight_3 = weight_4 = 1.0
    loss = torch.mean(weight_1 * diff_1) + torch.mean(weight_2 * diff_2) \
           + torch.mean(weight_3 * diff_3) + torch.mean(weight_4 * diff_4)
    return loss


def deformation_equality_loss(grid_A, grid_B):
    grid_eq = grid_A.permute([0, 3, 1, 2]) + grid_B.permute([0, 3, 1, 2])
    return torch.sqrt(torch.norm(grid_eq[:, 0, :, :])+torch.norm(grid_eq[:, 1, :, :])) #Frobenius norm
