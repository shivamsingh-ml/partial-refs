#!/usr/bin/env python3
"""
compute_gt_consensus.py

Combine multiple radiologist reports into consensus GT.
"""

import json
from pathlib import Path
from collections import Counter

def compute_consensus(gt_records: list) -> dict:
    """
    Combine multiple GT records (different radiologists) into consensus.
    
    For each finding:
    - List: Use majority vote (most common answer)
    - Enum: Use mode (most common value)
    - Object: Merge/average
    """
    
    if not gt_records:
        return {}
    
    case_id = gt_records[0]["case_id"]
    radiologists = [r["radiologist_id"] for r in gt_records]
    
    # Collect answers per question
    by_question = {}
    
    for gt_record in gt_records:
        for finding in gt_record.get("ground_truth", []):
            qid = finding["question_id"]
            answer = finding["answer_json"]
            
            if qid not in by_question:
                by_question[qid] = {
                    "category": finding["category"],
                    "question_text": finding["question_text"],
                    "answers": []  # All radiologists' answers
                }
            
            by_question[qid]["answers"].append(answer)
    
    # Compute consensus (majority vote for most fields)
    consensus_findings = []
    
    for qid, data in by_question.items():
        answers = data["answers"]
        
        # Simple approach: most common answer
        # (for complex objects, could use more sophisticated merging)
        answer_strs = [json.dumps(a, sort_keys=True) for a in answers]
        most_common = Counter(answer_strs).most_common(1)[0][0]
        consensus_answer = json.loads(most_common)
        
        # Track agreement
        agreement_ratio = len([a for a in answer_strs if a == most_common]) / len(answers)
        
        # Mark if disagreement detected
        confidence = "high" if agreement_ratio == 1.0 else "medium" if agreement_ratio >= 0.67 else "low"
        
        consensus_findings.append({
            "question_id": qid,
            "category": data["category"],
            "question_text": data["question_text"],
            "answer_json": consensus_answer,
            "confidence": confidence,
            "source": f"consensus_from_{len(answers)}_radiologists",
            "radiologists": radiologists,
            "agreement_ratio": agreement_ratio,
            "all_answers": answers  # Optional: keep all answers for analysis
        })
    
    return {
        "case_id": case_id,
        "radiologists": radiologists,
        "n_radiologists": len(radiologists),
        "ground_truth": consensus_findings
    }


def main():
    reports_dir = Path("dataset/reports")
    out_dir = Path("test_5/outputs/ground_truth")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by case ID
    by_case = {}
    for gt_file in out_dir.glob("*_doctor_*_ground_truth.jsonl"):
        # Parse: A003_doctor_1_ground_truth.jsonl
        case_id = gt_file.stem.split("_doctor_")[0]
        
        if case_id not in by_case:
            by_case[case_id] = []
        
        with open(gt_file) as f:
            by_case[case_id].append(json.loads(f.read()))
    
    # Compute consensus for each case
    for case_id, gt_records in by_case.items():
        consensus = compute_consensus(gt_records)
        
        # Save consensus GT
        consensus_file = out_dir / f"{case_id}_ground_truth.jsonl"
        with open(consensus_file, "w") as f:
            f.write(json.dumps(consensus) + "\n")
        
        print(f"[{case_id}] Computed consensus from {len(gt_records)} radiologists")
        
        # Report agreement
        for finding in consensus["ground_truth"][:3]:  # Show first 3
            agreement = finding.get("agreement_ratio", 0)
            print(f"  {finding['question_id']}: {agreement:.0%} agreement")


if __name__ == "__main__":
    main()