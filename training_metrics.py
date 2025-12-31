#!/usr/bin/env python3
"""
TRAINING METRICS & VERIFICATION SYSTEM
=======================================
Provides objective proof that the AI is learning.
Tracks training sessions, loss curves, and enables A/B comparison.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading

BASE_DIR = Path(__file__).parent
METRICS_DIR = BASE_DIR / "training_metrics"
HISTORY_FILE = METRICS_DIR / "training_history.json"


class TrainingMetricsTracker:
    """Track training metrics across sessions to verify learning is real"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        METRICS_DIR.mkdir(exist_ok=True)
        self.current_session = None
        self.history = self._load_history()
        self._initialized = True

    def _load_history(self) -> Dict:
        """Load training history from disk"""
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"sessions": [], "total_samples_trained": 0}
        return {"sessions": [], "total_samples_trained": 0}

    def _save_history(self):
        """Persist training history to disk"""
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def start_session(self, num_samples: int, model_type: str = "florence2") -> str:
        """Start a new training session"""
        self.current_session = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "model_type": model_type,
            "num_samples": num_samples,
            "epochs": [],
            "best_val_loss": float('inf'),
            "final_train_loss": None,
            "final_val_loss": None,
            "status": "in_progress"
        }
        self._save_session_progress()
        return self.current_session["id"]

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None):
        """Log metrics for an epoch"""
        if self.current_session is None:
            return

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "timestamp": datetime.now().isoformat()
        }

        if val_loss is not None:
            epoch_data["val_loss"] = val_loss
            if val_loss < self.current_session["best_val_loss"]:
                self.current_session["best_val_loss"] = val_loss

        self.current_session["epochs"].append(epoch_data)
        self._save_session_progress()

    def _save_session_progress(self):
        """Save in-progress session for crash recovery"""
        if self.current_session is None:
            return
        progress_file = METRICS_DIR / f"session_{self.current_session['id']}_progress.json"
        with open(progress_file, "w") as f:
            json.dump(self.current_session, f, indent=2)

    def end_session(self, success: bool = True, error_msg: str = None) -> str:
        """End training session and save to history"""
        if self.current_session is None:
            return None

        self.current_session["ended_at"] = datetime.now().isoformat()
        self.current_session["status"] = "completed" if success else "failed"

        if error_msg:
            self.current_session["error"] = error_msg

        if self.current_session["epochs"]:
            self.current_session["final_train_loss"] = self.current_session["epochs"][-1]["train_loss"]
            if "val_loss" in self.current_session["epochs"][-1]:
                self.current_session["final_val_loss"] = self.current_session["epochs"][-1]["val_loss"]

        # Add to history
        self.history["sessions"].append(self.current_session)
        if success:
            self.history["total_samples_trained"] += self.current_session["num_samples"]
        self._save_history()

        # Clean up progress file
        progress_file = METRICS_DIR / f"session_{self.current_session['id']}_progress.json"
        if progress_file.exists():
            progress_file.unlink()

        session_id = self.current_session["id"]
        self.current_session = None
        return session_id

    def get_learning_summary(self) -> Dict:
        """Get summary of all training to prove learning is real"""
        if not self.history["sessions"]:
            return {
                "total_sessions": 0,
                "learning_verified": False,
                "message": "No training has occurred yet. Label images and run training."
            }

        completed = [s for s in self.history["sessions"] if s["status"] == "completed"]

        if not completed:
            return {
                "total_sessions": len(self.history["sessions"]),
                "completed_sessions": 0,
                "learning_verified": False,
                "message": "Training attempted but never completed. Check for errors."
            }

        # Calculate improvement
        first_loss = completed[0].get("final_val_loss") or completed[0].get("final_train_loss", float('inf'))
        last_loss = completed[-1].get("final_val_loss") or completed[-1].get("final_train_loss", float('inf'))

        if first_loss != float('inf') and first_loss > 0:
            improvement = ((first_loss - last_loss) / first_loss * 100)
        else:
            improvement = 0

        best_loss = min(
            s.get("best_val_loss", s.get("final_train_loss", float('inf')))
            for s in completed
        )

        return {
            "total_sessions": len(self.history["sessions"]),
            "completed_sessions": len(completed),
            "total_samples_trained": self.history["total_samples_trained"],
            "first_session_loss": round(first_loss, 4) if first_loss != float('inf') else None,
            "latest_session_loss": round(last_loss, 4) if last_loss != float('inf') else None,
            "best_ever_loss": round(best_loss, 4) if best_loss != float('inf') else None,
            "improvement_percent": round(improvement, 2),
            "learning_verified": improvement > 5,  # At least 5% improvement
            "message": f"Model improved by {improvement:.1f}% over {len(completed)} sessions." if improvement > 0 else "Model needs more training."
        }

    def get_latest_session_epochs(self) -> List[Dict]:
        """Get epochs from the most recent completed session"""
        completed = [s for s in self.history["sessions"] if s["status"] == "completed"]
        if not completed:
            return []
        return completed[-1].get("epochs", [])


def generate_learning_report() -> str:
    """Generate a human-readable learning verification report"""
    tracker = TrainingMetricsTracker()
    summary = tracker.get_learning_summary()

    report = """
================================================================================
                     AI LEARNING VERIFICATION REPORT
================================================================================
"""

    if summary["total_sessions"] == 0:
        report += """
  STATUS: NO TRAINING DATA

  The AI has not been trained yet. To train:
  1. Label at least 5 images using active_learning.py
  2. Run: python train_florence2_pro.py
  3. Check this report again to verify learning
"""
    elif summary.get("completed_sessions", 0) == 0:
        report += """
  STATUS: TRAINING INCOMPLETE

  Training was attempted but never completed successfully.
  Check error logs and GPU memory availability.
"""
    else:
        verified = "YES - LEARNING CONFIRMED" if summary.get("learning_verified") else "NEEDS MORE DATA"
        report += f"""
  Training Sessions:      {summary['completed_sessions']}
  Total Samples Trained:  {summary['total_samples_trained']}

  First Session Loss:     {summary.get('first_session_loss', 'N/A')}
  Latest Session Loss:    {summary.get('latest_session_loss', 'N/A')}
  Best Ever Loss:         {summary.get('best_ever_loss', 'N/A')}

  Improvement:            {summary.get('improvement_percent', 0):+.1f}%
  Learning Verified:      {verified}

  {summary.get('message', '')}
"""

    report += """
================================================================================
"""

    return report


def check_finetuned_model_exists() -> bool:
    """Check if a fine-tuned model exists"""
    finetuned_path = BASE_DIR / "florence2_finetuned" / "final"
    return finetuned_path.exists()


def get_model_status() -> Dict:
    """Get current model status"""
    has_finetuned = check_finetuned_model_exists()
    tracker = TrainingMetricsTracker()
    summary = tracker.get_learning_summary()

    return {
        "finetuned_model_exists": has_finetuned,
        "using_model": "fine-tuned" if has_finetuned else "base (microsoft/Florence-2-large)",
        "training_summary": summary
    }


if __name__ == "__main__":
    print(generate_learning_report())
    print("\nModel Status:", get_model_status())
