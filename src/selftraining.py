import csv
import time

import _config_globals_ as globals
import _config_constants_ as cons
import _generic_commons_ as commons
import _load_model_test_iterate_ as lmti


def self_training():
    final_file = open('../dataset/' + lmti.get_file_prefix() + str(time.time()) + 'result.csv', 'w+')
    csv_result = csv.writer(final_file)
    csv_result.writerow(cons.CSV_HEADER)

    time_list = [time.time()]

    result = lmti.initial_run()
    print(result)
    csv_result.writerow(result)

    while lmti.ds.CURRENT_ITERATION < globals.NO_OF_ITERATION:
        if lmti.ds.CURRENT_ITERATION == 0:
            is_self_training = False
            # time_list_1 = [time.time()]

        else:
            is_self_training = True

        result = lmti.self_training_run(is_self_training)
        print (result)
        # if lmti.ds.CURRENT_ITERATION == 0:
            # time_list_1.append(time.time())
            # print(commons.temp_difference_cal(time_list_1))

        csv_result.writerow(result)

    final_file.close()
    time_list.append(time.time())
    print(commons.temp_difference_cal(time_list))


self_training()
