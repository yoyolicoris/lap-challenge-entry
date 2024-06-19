import argparse
from pathlib import Path
from sofar import read_sofa, write_sofa, Sofa

from const import LAP_MASK_INDEX


def main():
    parser = argparse.ArgumentParser(description="Simulate data for testing")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--regexp", type=str, default="*.sofa", help="Regular expression"
    )
    parser.add_argument("--lvl", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for f in input_dir.rglob(args.regexp):
        sofa = read_sofa(f)
        out_sofa = Sofa("SimpleFreeFieldHRIR")
        out_sofa.Data_IR = sofa.Data_IR[LAP_MASK_INDEX[args.lvl - 1]]
        out_sofa.SourcePosition = sofa.SourcePosition[LAP_MASK_INDEX[args.lvl - 1]]
        out_sofa.Data_SamplingRate = sofa.Data_SamplingRate
        write_sofa(output_dir / f.name, sofa)


if __name__ == "__main__":
    main()
