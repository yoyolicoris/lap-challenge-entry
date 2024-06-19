from spaudiopy.sph import sh_matrix, inverse_sht
from spaudiopy.utils import sph2cart, cart2sph
import numpy as np
from typing import Tuple
from scipy.interpolate import griddata
from pyfar import Signal, Coordinates
import argparse
from pathlib import Path
from pyfar.io import read_sofa
from pyfar import rad2deg
from sofar import write_sofa, Sofa

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


def upsample_v1(
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

    est_toa = toa(
        hrir=hrir.time,
        xyz=input_coords.cartesian,
        sr=sr,
        **toa_kwargs,
    )
    shifts = est_toa

    hrtf = hrir.freq_raw
    freqs = hrir.frequencies
    aligned_hrtf = hrtf * np.exp(2j * np.pi * freqs / sr * shifts[..., None])

    sph_coords = input_coords

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
        num_aug_points = np.count_nonzero(unique_indices >= input_xyz.shape[0])

        print(
            f"Number of augmented grid points: {num_aug_points}, total grid points: {input_xyz.shape[0]}"
        )
        if num_aug_points > 0:
            aug_xyz = np.concatenate(
                (input_xyz, input_xyz * np.array([1, -1, 1])), axis=0
            )
            input_xyz = aug_xyz[unique_indices]
            aug_aligned_hrtf = np.concatenate(
                (aligned_hrtf, np.fliplr(aligned_hrtf)), axis=0
            )
            sph_coords = Coordinates(
                points_1=input_xyz[:, 0],
                points_2=input_xyz[:, 1],
                points_3=input_xyz[:, 2],
            )
            aligned_hrtf = aug_aligned_hrtf[unique_indices]

    sparse_sph = sht_lstsq_reg(
        aligned_hrtf.reshape(aligned_hrtf.shape[0], -1),
        sph_order,
        sph_coords.azimuth,
        sph_coords.colatitude,
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
        pred_shifts = pred_toa
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


def main():
    parser = argparse.ArgumentParser(description="Upsample HRIRs")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--sph-order", type=int, default=2, help="Spherical harmonics order"
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="Regularization factor")
    parser.add_argument(
        "--lr-augment", action="store_true", help="Left-right augmentation"
    )
    parser.add_argument(
        "--use-rigid-toa", action="store_true", help="Use rigid toa alignment"
    )
    parser.add_argument(
        "--oversampling", type=int, default=10, help="Oversampling factor"
    )
    parser.add_argument(
        "--theta", type=float, default=8, help="Exponent for toa weighting"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--ref",
        type=str,
        default="/Users/ycy/Documents/lap-challenge/SONICOM/P0001-P0010/P0001/HRTF/HRTF/48kHz/P0001_FreeFieldCompMinPhase_48kHz.sofa",
        help="Reference sofa file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    _, ref_coords, _ = read_sofa(args.ref)

    for f in input_dir.glob("*.sofa"):
        hrir, coords, _ = read_sofa(f)
        pred_hrir, _ = upsample_v1(
            hrir,
            coords,
            ref_coords,
            sph_order=args.sph_order,
            eps=args.eps,
            lr_augment=args.lr_augment,
            use_rigid_toa=args.use_rigid_toa,
            oversampling=args.oversampling,
            theta=args.theta,
            verbose=args.verbose,
        )

        sofa = Sofa("SimpleFreeFieldHRIR")
        sofa.Data_IR = pred_hrir.time
        sofa.SourcePosition = rad2deg(ref_coords.spherical_elevation)
        sofa.Data_SamplingRate = hrir.sampling_rate
        write_sofa(output_dir / f.name, sofa)


if __name__ == "__main__":
    main()
