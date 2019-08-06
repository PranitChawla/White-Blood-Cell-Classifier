import inference 
PATH_TO_FROZEN_GRAPH="/home/pranit/Desktop/white_blood_cell_project/white_blood_cell_graph/frozen_inference_graph.pb"
PATH_TEST="/home/pranit/Desktop/white_blood_cell_project/BCCD_Dataset/BCCD/JPEGImages"
PATH_LABELS="/home/pranit/Desktop/white_blood_cell_project/labels.pbtxt"
inference.create_results(PATH_TO_FROZEN_GRAPH,PATH_TEST,PATH_LABELS)