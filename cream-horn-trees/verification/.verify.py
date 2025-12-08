import json
LABELS_TEST_FILE = 'labels_test.csv'
TREES_TEST_FILE = 'trees_test.csv'
labels_test_df = pd.read_csv(LABELS_TEST_FILE)
trees_test_df = pd.read_csv(TREES_TEST_FILE)
trees_test_df['tree'] = trees_test_df['tree'].map(lambda x: eval(x))

    
def rate_solution_test(solution_fn, labels_df, trees_df):
    images_path = labels_df["image_path"].unique()

    try: 
        predictions = solution_fn(images_path)
    except Exception as e:
        print(json.dumps({
            "passed": False,
            "message": f"Error running the solution function: {str(e)}"
        }))
        return
    
    try:
        evaluation = evaluate_predictions(predictions, labels_df, trees_df)
    except Exception as e:
        print(json.dumps({
            "passed": False,
            "message": f"Error during evaluation: {str(e)}"
        }))

        return
    
    if evaluation['total_score'] >= 80:
        print(json.dumps({
            "passed": True,
            "message": f"Points {evaluation['total_score']:.2f} pkt: F1-score: {evaluation['detections_f1'] * 100:.2f}%, {evaluation['detections_f1'] * 60:.2f} pkt; Tree accuracy: {evaluation['trees_accuracy'] * 100:.2f}%, {evaluation['trees_accuracy'] * 40:.2f} pkt"
        }))

    else:
        print(json.dumps({
            "passed": False,
            "message": f"Points {evaluation['total_score']:.2f} pkt: F1-score: {evaluation['detections_f1'] * 100:.2f}%, {evaluation['detections_f1'] * 60:.2f} pkt; Tree accuracy: {evaluation['trees_accuracy'] * 100:.2f}%, {evaluation['trees_accuracy'] * 40:.2f} pkt"
        }))

rate_solution_test(your_solution, labels_test_df, trees_test_df)