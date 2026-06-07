""""""

"""
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: This program evaluate KITTI odometry result
"""

import argparse
from eval.kitti_odometry import KittiEvalOdom
from eval.seven_scenes_odometry import SevenScenesEvalOdom
from eval.euroc_odometry import EuRoCEvalOdom


def argument_parsing():
    """Argument parsing

    Returns:
        args (args): arguments
    """
    parser = argparse.ArgumentParser(description="KITTI Odometry evaluation")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument("--result", type=str, required=True, help="Result directory")
    parser.add_argument(
        "--gt",
        type=str,
        default=r"C:/workspace/data/kitti/poses",  # "dataset/kitti_odom/gt_poses/",
        help="GT Pose directory containing gt pose txt files",
    )
    parser.add_argument(
        "--align",
        type=str,
        choices=["scale", "scale_7dof", "7dof", "6dof"],
        default=None,
        help="alignment type",
    )
    parser.add_argument(
        "--seqs", nargs="+", help="sequences to be evaluated", default=None
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # argument parsing
    args = argument_parsing()

    # initialize evaluation tool
    if args.dataset == "kitti":
        eval_tool = KittiEvalOdom()
    elif args.dataset == "7scenes":
        eval_tool = SevenScenesEvalOdom(lengths=[x for x in range(1, 11, 1)])
        args.gt = "C:/workspace/data/7scenes"
    elif args.dataset == "euroc":
        eval_tool = EuRoCEvalOdom(lengths=[x for x in range(10, 110, 10)])
        args.gt = "G:/data/euroc"

    continue_flag = input("Evaluate result in [{}]? [y/n]".format(args.result))
    if continue_flag == "y":
        eval_tool.eval(
            args.gt,
            args.result,
            alignment=args.align,
            seqs=args.seqs,
        )
    else:
        print("Double check the path!")
