import numpy as np

def compute_f05_score(ground_truth_dict, predictions_dict):
    """
    ground_truth_dict: dict of {source1_entity_id: set(matched_s2_s3_ids)}
    predictions_dict: dict of {source1_entity_id: set(predicted_s2_s3_ids)}
    If a source1 entity is a singleton, the set should be empty.
    """
    f05_scores = []
    
    # Evaluate for all entities in ground truth
    for s1_id, true_matches in ground_truth_dict.items():
        pred_matches = predictions_dict.get(s1_id, set())
        
        if len(true_matches) == 0 and len(pred_matches) == 0:
            f05_scores.append(1.0)
            continue
        
        if len(pred_matches) == 0 or len(true_matches) == 0:
            f05_scores.append(0.0)
            continue
            
        intersection = len(true_matches.intersection(pred_matches))
        p = intersection / len(pred_matches)
        r = intersection / len(true_matches)
        
        if p == 0 or r == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)
            f05_scores.append(f05)
            
    return np.mean(f05_scores)
