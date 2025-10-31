import torch

def compare_model_keys(pth1, pth2):
    state1 = torch.load(pth1, map_location="cpu")
    state2 = torch.load(pth2, map_location="cpu")

    # 有些 checkpoint 会多包一层 dict
    if "state_dict" in state1:
        state1 = state1["state_dict"]
    if "state_dict" in state2:
        state2 = state2["state_dict"]

    keys1 = set(state1.keys())
    keys2 = set(state2.keys())

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    print(f"共有参数: {len(common)}")
    print(f"仅在 {pth1} 中的参数 ({len(only_in_1)}):")
    for k in sorted(only_in_1):
        print("  ", k)

    print(f"\n仅在 {pth2} 中的参数 ({len(only_in_2)}):")
    for k in sorted(only_in_2):
        print("  ", k)

    if not only_in_1 and not only_in_2:
        print("\n✅ 两个模型的结构完全一致！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pth1", help="第一个 pth 文件路径")
    parser.add_argument("pth2", help="第二个 pth 文件路径")
    args = parser.parse_args()

    compare_model_keys(args.pth1, args.pth2)