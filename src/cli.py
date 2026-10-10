import argparse
from src.pipeline import run

def main():
    parser = argparse.ArgumentParser(description="Flood Mapper")
    subparsers = parser.add_subparsers(dest="command")
    
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--bbox", type=str, required=True, help="minx,miny,maxx,maxy")
    run_parser.add_argument("--date", type=str, required=True, help="Flood date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    if args.command == "run":
        bbox = [float(x) for x in args.bbox.split(",")]
        run(bbox, args.date)

if __name__ == "__main__":
    main()
