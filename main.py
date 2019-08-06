import create_tf_records
sets="val"
label_dict={"WBC":1,"RBC":2,"Platelets":3}
create_tf_records.main(sets,label_dict)