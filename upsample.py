from spaudiopy.sph import sh_matrix, inverse_sht
from spaudiopy.utils import sph2cart, cart2sph
import numpy as np
from typing import Tuple
from scipy.interpolate import griddata
from pyfar import Signal, Coordinates

from toa import toa, get_rigid_params, toa_model


def sht_lstsq_reg(f, N_sph, azi, colat, sh_type, Y_nm=None, eps: float = 1e-5):
    if f.ndim == 1:
        f = f[:, np.newaxis]  # upgrade to handle 1D arrays
    if Y_nm is None:
        Y_nm = sh_matrix(N_sph, azi, colat, sh_type)
    reg = sum(([1 + n * (n + 1)] * (2 * n + 1) for n in range(N_sph + 1)), start=[])
    return np.linalg.solve(Y_nm.T.conj() @ Y_nm + eps * np.diag(reg), Y_nm.T.conj() @ f)


def time_align_upsample(
    hrir: np.ndarray,
    input_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sph_order: int,
    sr: int,
    eps: float = 1e-5,
    use_rigid_toa: bool = False,
    use_linear: bool = False,
    **toa_kwargs,
):
    input_az, input_col, input_r = input_grid
    target_az, target_col, target_r = target_grid
    input_xyz = np.stack(sph2cart(input_az, input_col, input_r), 1)

    est_toa = toa(
        hrir=hrir,
        xyz=input_xyz,
        sr=sr,
        **toa_kwargs,
    )

    shifts = est_toa - est_toa.min(0)

    hrtf = np.fft.rfft(hrir, axis=-1)
    freqs = np.fft.rfftfreq(hrir.shape[-1], 1 / sr)
    aligned_hrtf = hrtf * np.exp(2j * np.pi * freqs / sr * shifts[..., None])
    if use_linear:
        aligned_hrir = np.fft.irfft(aligned_hrtf, axis=-1)

        az = np.concatenate((input_az - 2 * np.pi, input_az, input_az + 2 * np.pi) * 3)
        col = np.concatenate(
            (-input_col,) * 3 + (input_col,) * 3 + (2 * np.pi - input_col,) * 3
        )
        v = np.concatenate((aligned_hrir,) * 9, axis=0).reshape(
            9 * aligned_hrir.shape[0], -1
        )

        pred_hrir = griddata(
            (az, col), v, (target_az, target_col), method="linear"
        ).reshape(-1, *aligned_hrir.shape[1:])
        pred_hrtf = np.fft.rfft(pred_hrir, axis=-1)
    else:
        sparse_sph = sht_lstsq_reg(
            aligned_hrtf.reshape(aligned_hrtf.shape[0], -1),
            sph_order,
            input_az,
            input_col,
            "complex",
            eps=eps,
        )

        pred_hrtf = inverse_sht(sparse_sph, target_az, target_col, "complex").reshape(
            -1, *aligned_hrtf.shape[1:]
        )

    # upsample toa
    if use_rigid_toa:
        rigid_params = get_rigid_params(est_toa, input_xyz, sr, verbose=True)
        target_xyz = np.stack(sph2cart(target_az, target_col, target_r), 1)
        pred_toa = toa_model(
            target_xyz,
            sr=sr,
            **rigid_params,
        )
        pred_shifts = pred_toa - est_toa.min(0)
    else:
        pred_shifts = inverse_sht(
            sht_lstsq_reg(
                shifts,
                sph_order,
                input_az,
                input_col,
                "real",
                eps=eps,
            ),
            target_az,
            target_col,
            "real",
        )

    upsampled_hrtf = pred_hrtf * np.exp(
        -2j * np.pi * freqs / sr * pred_shifts[..., None]
    )
    upsampled_hrir = np.fft.irfft(upsampled_hrtf, axis=-1)

    return upsampled_hrir, pred_shifts


def lv4_upsample(
    hrir: Signal,
    input_coords: Coordinates,
    target_coords: Coordinates,
    sph_order: int = 2,
    eps: float = 1e-5,
    lr_augment: bool = False,
    use_rigid_toa: bool = False,
    **toa_kwargs,
):
    sr = hrir.sampling_rate
    input_xyz = input_coords.cartesian

    if lr_augment:
        # augment input grid
        aug_sph_coords = (
            np.round(
                np.vstack(
                    (
                        np.concatenate(
                            (input_coords.azimuth, 2 * np.pi - input_coords.azimuth)
                        ),
                        np.tile(input_coords.colatitude, 2),
                    )
                ).T
                / np.pi
                * 180
            ).astype(int)
            % 360
        )
        _, unique_indices = np.unique(aug_sph_coords, return_index=True, axis=0)
        print(_)
        num_aug_points = np.count_nonzero(unique_indices >= input_xyz.shape[0])

        print(
            f"Number of augmented grid points: {num_aug_points}, total grid points: {input_xyz.shape[0]}"
        )
        if num_aug_points > 0:
            aug_xyz = np.concatenate(
                (input_xyz, input_xyz * np.array([1, -1, 1])), axis=0
            )
            input_xyz = aug_xyz[unique_indices]
            aug_hrir = np.concatenate((hrir.time, np.fliplr(hrir.time)), axis=0)
            hrir = Signal(aug_hrir[unique_indices], sr)
            input_coords = Coordinates(
                points_1=input_xyz[:, 0],
                points_2=input_xyz[:, 1],
                points_3=input_xyz[:, 2],
            )

    est_toa = toa(
        hrir=hrir.time,
        xyz=input_coords.cartesian,
        sr=sr,
        **toa_kwargs,
    )
    offset = est_toa.min(0)
    shifts = est_toa - offset

    hrtf = hrir.freq_raw
    freqs = hrir.frequencies
    aligned_hrtf = hrtf * np.exp(2j * np.pi * freqs / sr * shifts[..., None])

    sparse_sph = sht_lstsq_reg(
        aligned_hrtf.reshape(aligned_hrtf.shape[0], -1),
        sph_order,
        input_coords.azimuth,
        input_coords.colatitude,
        "complex",
        eps=eps,
    )
    pred_hrtf = inverse_sht(
        sparse_sph, target_coords.azimuth, target_coords.colatitude, "complex"
    ).reshape(-1, *aligned_hrtf.shape[1:])

    if use_rigid_toa:
        rigid_params = get_rigid_params(
            est_toa, input_coords.cartesian, sr, verbose=True
        )
        pred_toa = toa_model(
            target_coords.cartesian,
            sr=sr,
            **rigid_params,
        )
        pred_shifts = pred_toa - offset
    else:
        pred_shifts = inverse_sht(
            sht_lstsq_reg(
                shifts,
                sph_order,
                input_coords.azimuth,
                input_coords.colatitude,
                "real",
                eps=eps,
            ),
            target_coords.azimuth,
            target_coords.colatitude,
            "real",
        )

    upsampled_hrtf = pred_hrtf * np.exp(
        -2j * np.pi * freqs / sr * pred_shifts[..., None]
    )
    upsampled_hrir = np.fft.irfft(upsampled_hrtf, axis=-1)

    return (
        Signal(
            upsampled_hrir,
            sr,
        ),
        pred_shifts,
    )
