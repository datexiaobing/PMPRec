from data_helper import DataHelper
if __name__ == '__main__':
    helper = DataHelper()
    # step 1
    helper.generate_node_id_to_class_file()
    # step 2
    helper.generate_new_id_ubFile()
    # step 3
    helper.generate_all_edges_file()
    # step 4
    helper.generate_train_test_data()
    # step 5
    helper.generate_train_test_data_for_herec()
    # step 6
    helper.generate_train_edges_file()
    # step 7
    helper.generate_meta_path_3(save_base_path= '../new/train/',train_count=3)
    # step 8
    helper.generate_meta_path_5(save_base_path= '../new/train/',train_count=3)
