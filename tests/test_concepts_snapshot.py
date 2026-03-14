import sys
import os
import json
import chess
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from app.concepts import extract_board_features

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Starting
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", # Sicilian
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5", # Italian
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", # Complex
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", # Simple Endgame
    "2r3k1/1p3p1p/p2p2p1/3P4/P1P1R3/1P6/5PPP/6K1 b - - 0 24", # Rook endgame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4" # Four Knights
]

SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "concepts_snapshot.json")

def generate_snapshots():
    snapshots = {}
    for fen in FENS:
        board = chess.Board(fen)
        features = extract_board_features(board)
        snapshots[fen] = features
    
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(snapshots, f, indent=4)
    print(f"✅ Saved baseline snapshots to {SNAPSHOT_FILE}")

def verify_snapshots():
    if not os.path.exists(SNAPSHOT_FILE):
        print(f"❌ Snapshot file not found at {SNAPSHOT_FILE}. Run with --save first.")
        sys.exit(1)
        
    with open(SNAPSHOT_FILE, "r") as f:
        baseline = json.load(f)
        
    success = True
    for fen in FENS:
        board = chess.Board(fen)
        current = extract_board_features(board)
        
        if fen not in baseline:
             print(f"❌ FEN not in baseline: {fen}")
             success = False
             continue
             
        # Convert to JSON string and back for exact structure match (floats might be formatted differently)
        current_stripped = json.loads(json.dumps(current))
        base_stripped = baseline[fen]
        
        if current_stripped != base_stripped:
             print(f"❌ MISMATCH for FEN: {fen}")
             print("Differences found. Use a diff tool to compare.")
             success = False
        else:
             print(f"✅ PASSED: {fen[:40]}...")

    if success:
        print("🎉 ALL SNAPSHOTS MATCH PERFECTLY!")
    else:
        print("🚨 REGRESSION DETECTED!")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save current output as baseline")
    parser.add_argument("--verify", action="store_true", help="Verify against baseline")
    args = parser.parse_args()
    
    if args.save:
        generate_snapshots()
    elif args.verify:
        verify_snapshots()
    else:
        parser.print_help()
