import os, sys
import argparse
from preprocess.split_datasets import main as split_concept
from preprocess.split_datasets_que import main as split_question
from preprocess import data_proprocess, process_raw_data

dname2paths = {
    "assist2009": "/root/workcopy/datasets/data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2012": "/root/workcopy/datasets/data/assist2012/2012-2013-data-with-predictions-4-final.csv",
    "assist2015": "/root/workcopy/datasets/data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "/root/workcopy/datasets/data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "/root/workcopy/datasets/data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "/root/workcopy/datasets/data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "/root/workcopy/datasets/data/nips_task34/train_task_3_4.csv",
    "poj": "/root/workcopy/datasets/data/poj/poj_log.csv",
    "slepemapy": "/root/workcopy/datasets/data/slepemapy/answer.csv",
    "assist2017": "/root/workcopy/datasets/data/assist2017/anonymized_full_release_competition_dataset.csv",
    "junyi2015": "/root/workcopy/datasets/data/junyi2015/junyi_ProblemLog_original.csv",
    "ednet": "/root/workcopy/datasets/data/ednet/",
    "ednet5w": "/root/workcopy/datasets/data/ednet/",
    "peiyou": "/root/workcopy/datasets/data/peiyou/grade3_students_b_200.csv",
    "comparch2022": "/root/workcopy/datasets/data/comparch2022/comparch_220947_data.csv"
}
configf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./configs/data_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="comparch2022")
    parser.add_argument("-f","--file_path", type=str, default="/root/workcopy/datasets/data/peiyou/grade3_students_b_200.csv")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)

    # process raw data
    if args.dataset_name=="peiyou":
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    print(f"dname: {dname}, writef: {writef}")
    # split
    os.system("rm " + dname + "/*.pkl")

    #for concept level model
    split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    print("="*100)

    #for question level model
    split_question(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)

