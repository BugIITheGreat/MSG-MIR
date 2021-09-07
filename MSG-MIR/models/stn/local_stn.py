import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DownBlock, Conv, ResnetTransformer, get_activation, TransConv
from .stn_losses import smoothness_loss, deformation_equality_loss

sampling_align_corners = False
sampling_mode = 'bilinear'

# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 32, 64, 64, 128, 128, 256], }
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [256, 128, 128, 64, 64, 32, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 5, }
# indicate the time for output the intact affine parameters.
convs_for_intact = {'A': 7, }
# control the contribution of intact feature and local feature.
para_for_local = {'A': 0.9, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }

affine_dimentions = {'A': 6, }


class LocalNet(nn.Module):
    def __init__(self, nc_a, nc_b, cfg, height, width, init_func, init_to_identity):
        super(LocalNet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        self.h, self.w = height, width
        self.convs_for_intact = convs_for_intact[cfg]

        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_norm=True))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1

        actIntact = get_activation(activation='relu')
        self.outputIntact = nn.Sequential(
            nn.Linear(ndf[cfg][self.convs_for_intact - 1] *
                      (self.h // 2 ** (self.convs_for_intact - 1)) *
                      (self.w // 2 ** (self.convs_for_intact - 1)),
                      ndf[cfg][self.convs_for_intact - 1], bias=True),
            actIntact,
            nn.Linear(ndf[cfg][self.convs_for_intact - 1], affine_dimentions[cfg], bias=True))

        self.outputIntact[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.outputIntact[-1].bias.data.zero_()

        if use_down_resblocks[cfg]:
            self.c1 = Conv(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=act,
                         init_fun=init_func, use_norm=True, use_resnet=True))
            setattr(self, 'output_{}'.format(conv_num),
                    Conv(out_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                         init_func=('zeros' if init_to_identity else init_func), activation=act,
                         use_norm=False)
                    )
            # ------------- Deformation Field TransposeConv Block
            setattr(self, 'field_transconv_{}'.format(conv_num),
                    TransConv(2, 2, 3, 2, 0, use_resnet=True, bias=True,
                              init_func=('zeros' if init_to_identity else init_func), activation=act,
                              use_norm=False)
                    )
            if refine_output[cfg]:
                setattr(self, 'refine_{}'.format(conv_num),
                        nn.Sequential(ResnetTransformer(out_nf, 1, init_func),
                                      Conv(out_nf, out_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                           activation=act,
                                           use_norm=False)
                                      )
                        )
            else:
                setattr(self, 'refine_{}'.format(conv_num), lambda x: x)
            in_nf = out_nf
            conv_num -= 1

    def forward(self, img_a, img_b):
        use_transpose_conv_in_fields = False
        para_for_multiscale = 0.9
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1

        tus = skip_vals['down_{}'.format(self.convs_for_intact)]
        # print(str(tus.shape) + "tus_shape")
        intact_x = tus.view(tus.size(0), -1)
        # print(str(intact_x.shape) + "intact_x_shape")
        # print(self.outputIntact)
        dtheta_for_intact = self.outputIntact(intact_x)

        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        deform_scale_output = {}
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            x = getattr(self, 'refine_{}'.format(conv_num))(x)
            deform_scale_output[conv_num] = getattr(self, 'output_{}'.format(conv_num))(x)
            if use_transpose_conv_in_fields is False:
                if conv_num is self.nup_blocks:
                    def_for_local = deform_scale_output[conv_num]
                else:
                    def_for_local = para_for_multiscale * F.interpolate(def_for_local,
                                                                        (deform_scale_output[conv_num].shape[2],
                                                                         deform_scale_output[conv_num].shape[3]),
                                                                        mode='bilinear') \
                                    + deform_scale_output[conv_num]
            else:
                if conv_num is self.nup_blocks:
                    def_for_local = deform_scale_output[conv_num]
                else:
                    ppr = getattr(self, 'field_transconv_{}'.format(conv_num))(def_for_local)
                    ppr = F.interpolate(ppr,
                                        (deform_scale_output[conv_num].shape[2],
                                         deform_scale_output[conv_num].shape[3]),
                                        mode='bilinear')
                    def_for_local = para_for_multiscale * ppr + deform_scale_output[conv_num]


            conv_num -= 1
        # x = self.outputLocal(x)
        return dtheta_for_intact, def_for_local, deform_scale_output


class LocalSTN(nn.Module):
    """This class is generates and applies the deformable transformation on the input images."""

    def __init__(self, in_channels_a, in_channels_b, height, width, cfg, init_func, stn_bilateral_alpha,
                 init_to_identity, multi_resolution_regularization):
        super(LocalSTN, self).__init__()
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_offsets = LocalNet(self.in_channels_a, self.in_channels_b, cfg, height, width,
                                    init_func, init_to_identity).to(self.device)
        self.identity_grid = self.get_identity_grid()
        self.identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(self.device)
        if affine_dimentions[cfg] is 8:
            self.identity_theta = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float).to(self.device)
        if affine_dimentions[cfg] is 6:
            self.identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(self.device)
        elif affine_dimentions[cfg] is 4:
            self.justian_matrix = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.float).unsqueeze(0)
        # self.transfer_matrix = torch.tensor(self.justian_matrix, dtype=torch.float).to(self.device)
        self.alpha = stn_bilateral_alpha
        self.multi_resolution_regularization = multi_resolution_regularization
        self.para_for_local = para_for_local[cfg]

    def get_identity_grid(self):
        """Returns a sampling-grid that represents the identity transformation."""
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def get_homography_grid(self, matrix):
        # matrix = torch.cat((matrix, torch.ones([1, 1]).to(matrix.device)), dim=1)
        matrix = matrix.view(3, 3)
        identity_grid = self.get_identity_grid()
        height = identity_grid.shape[2]
        width = identity_grid.shape[3]
        identity_grid = identity_grid.view(1, 2, -1).to(matrix.device)
        com = torch.ones([1, 1, identity_grid.shape[-1]]).to(identity_grid.device)
        identity_grid = torch.cat((identity_grid, com), dim=1).squeeze(0)
        homo_grid = torch.matmul(matrix, identity_grid)
        with torch.no_grad():
            homo_grid[0, :] = torch.div(homo_grid[0, :], homo_grid[2, :])
            homo_grid[1, :] = torch.div(homo_grid[1, :], homo_grid[2, :])
        torch.set_grad_enabled(True)
        return homo_grid[0:2, :].view(1, 2, height, width)

    def get_affine_grid(self, matrix):
        matrix = matrix.view(2, 3)
        identity_grid = self.get_identity_grid()
        height = identity_grid.shape[2]
        width = identity_grid.shape[3]
        identity_grid = identity_grid.view(1, 2, -1).to(matrix.device)
        com = torch.ones([1, 1, identity_grid.shape[-1]]).to(identity_grid.device)
        identity_grid = torch.cat((identity_grid, com), dim=1).squeeze(0)
        aff_suf = torch.tensor([0, 0, 1], dtype=torch.float).unsqueeze(0).to(self.device)
        matrix = torch.cat((matrix, aff_suf), dim=0)
        affine_grid = torch.matmul(matrix, identity_grid)
        return affine_grid[0:2, :].view(1, 2, height, width)

    def get_grid(self, img_a, img_b, return_offsets_only=False):
        """Return the predicted sampling grid that aligns img_a with img_b."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)
        # Get Deformation Field
        b_size = img_a.size(0)
        all_offsets = self.all_offsets(img_a, img_b)

        dtheta_for_intact = all_offsets[0]
        theta_for_intact = dtheta_for_intact + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
        if dtheta_for_intact.shape[-1] == 6:
            theta_for_intact = dtheta_for_intact + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
            trans_grid = self.get_affine_grid(theta_for_intact)
        elif dtheta_for_intact.shape[-1] == 8:
            dtheta_for_intact = torch.cat((dtheta_for_intact, torch.ones([1, 1]).to(img_a.device)), dim=1)
            theta_for_intact = dtheta_for_intact + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
            trans_grid = self.get_homography_grid(theta_for_intact)

        deformation = all_offsets[1]
        deformation_upsampled = deformation
        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(deformation, (self.oh, self.ow), mode=sampling_mode,
                                                  align_corners=sampling_align_corners)
        if return_offsets_only:
            resampling_grid = deformation_upsampled.permute([0, 2, 3, 1])
        else:
            resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])
            if dtheta_for_intact.shape[-1] < 6:
                resampling_grid_intact = F.affine_grid(theta_for_intact.view(-1, 2, 3), img_a.size())
            else:
                resampling_grid_intact = trans_grid.permute([0, 2, 3, 1])

            kkp = resampling_grid
            resampling_grid = resampling_grid_intact + self.para_for_local * resampling_grid

        ksa = all_offsets[2]
        return resampling_grid_intact

    def forward(self, img_a, img_b, apply_on=None):
        """
        Predicts the spatial alignment needed to align img_a with img_b. The spatial transformation will be applied
        on the tensors passed by apply_on (if apply_on is None then the transformation will be applied on img_a).

            :param img_a: the source image.
            :param img_b: the target image.
            :param apply_on: the geometric transformation can be applied on different tensors provided by this list.
                        If not set, then the transformation will be applied on img_a.
            :return: a list of the warped images (matching the order they appeared in apply on), and the regularization term
                        calculated for the predicted transformation."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)
        # Get Deformation Field
        b_size = img_a.size(0)
        all_offsets = self.all_offsets(img_a, img_b)

        dtheta_for_intact = all_offsets[0]
        if dtheta_for_intact.shape[-1] < 6:
            # dtheta_for_intact = dtheta_for_intact * self.transfer_matrix
            theta_for_intact = dtheta_for_intact + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
        else:
            if dtheta_for_intact.shape[-1] == 6:
                theta_for_intact = dtheta_for_intact + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
                trans_grid = self.get_affine_grid(theta_for_intact)
            elif dtheta_for_intact.shape[-1] == 8:
                dtheta_for_intact = torch.cat((dtheta_for_intact, torch.ones([1, 1]).to(img_a.device)), dim=1)
                theta_for_intact = dtheta_for_intact + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
                trans_grid = self.get_homography_grid(dtheta_for_intact)

        deformation = all_offsets[1]
        deformation_upsampled = deformation
        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(deformation, (self.oh, self.ow), mode=sampling_mode)
        resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])
        # Wrap image wrt to the deformation field
        if apply_on is None:
            apply_on = [img_a]
        warped_images = []
        for img in apply_on:
            if dtheta_for_intact.shape[-1] < 6:
                resampling_grid_intact = F.affine_grid(theta_for_intact.view(-1, 2, 3), img_a.size())
            else:
                resampling_grid_intact = trans_grid.permute([0, 2, 3, 1])
            resampling_grid = (1 - self.para_for_local) * resampling_grid_intact + self.para_for_local * resampling_grid
            warped_images.append(F.grid_sample(img, resampling_grid, mode=sampling_mode, padding_mode='zeros',
                                               align_corners=sampling_align_corners))
        # Calculate STN regularization term
        reg_term = self._calculate_regularization_term(deformation, warped_images[0])
        # return warped_images, reg_term, resampling_grid
        return warped_images, reg_term

    def _calculate_regularization_term(self, deformation, img):
        """Calculate the regularization term of the predicted deformation.
        The regularization may-be applied to different resolution for larger images."""
        dh, dw = deformation.size(2), deformation.size(3)
        img = None if img is None else img.detach()
        reg = 0.0
        factor = 1.0
        for i in range(self.multi_resolution_regularization):
            if i != 0:
                deformation_resized = F.interpolate(deformation, (dh // (2 ** i), dw // (2 ** i)), mode=sampling_mode,
                                                    align_corners=sampling_align_corners)
                img_resized = F.interpolate(img, (dh // (2 ** i), dw // (2 ** i)), mode=sampling_mode,
                                            align_corners=sampling_align_corners)
            elif deformation.size()[2::] != img.size()[2::]:
                deformation_resized = deformation
                img_resized = F.interpolate(img, deformation.size()[2::], mode=sampling_mode,
                                            align_corners=sampling_align_corners)
            else:
                deformation_resized = deformation
                img_resized = img
            reg += factor * smoothness_loss(deformation_resized, img_resized, alpha=self.alpha)
            factor /= 2.0
        return reg
