import sys
import os

# Add project root to path as fallback
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from maia2 import model, inference
print("Maia2 imported successfully.")

print("Loading model...")
model = model.from_pretrained(type="rapid", device="cpu")
print("Model loaded.")

print("Preparing inference...")
prepared = inference.prepare()
print("Inference prepared.")

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
elo_self = 1500
elo_oppo = 1500

print(f"Running inference for FEN: {fen}")
move_probs, win_prob = inference.inference_each(model, prepared, fen, elo_self, elo_oppo)

print(f"Win probability: {win_prob}")
if move_probs:
    best_move = max(move_probs, key=move_probs.get)
    print(f"Best move: {best_move} (Prob: {move_probs[best_move]})")
    print(f"Type of move key: {type(best_move)}")
    print(f"Sample keys: {list(move_probs.keys())[:5]}")
else:
    print("No move probabilities returned.")
