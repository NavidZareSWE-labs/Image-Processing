from visualize import plot_result_s0
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

os.makedirs('output/utils', exist_ok=True)


def check_embedding_feasibility(sm_row, sm_col, lg_row, lg_col):
    min_required_dim = sm_row + sm_col - 1
    target_too_small = lg_row < min_required_dim or lg_col < min_required_dim
    return [min_required_dim, target_too_small]


def embed_diamond(SM, LG):
    sm_row, sm_col = SM.shape
    lg_row, lg_col = LG.shape

    min_required_dim, target_too_small = check_embedding_feasibility(
        sm_row, sm_col, lg_row, lg_col)

    if target_too_small:
        return "Impossible"

    lg_out = LG.copy().astype(SM.dtype)

    # Centre the diamond inside L
    row_offset = (lg_row - min_required_dim) // 2
    col_offset = (lg_col - min_required_dim) // 2

    for i in range(sm_row):
        for j in range(sm_col):
            lg_mat_row = (sm_col - 1) - j + i + row_offset
            lg_mat_col = i + j + col_offset
            lg_out[lg_mat_row, lg_mat_col] = SM[i, j]

    return lg_out


def run_utils():
    print("\n" + "=" * 60)
    print("SECTION 0 : Random Diamond Pattern Embedding")
    print("=" * 60)

    # ---- Case 1: 5x5 S into 9x9 L (exact fit) ----
    SM1 = np.arange(1, 26).reshape(5, 5)
    LG1 = np.zeros((9, 9), dtype=int)
    result = embed_diamond(SM1, LG1)
    print("\n[Example 1] S (5x5) into L (9x9 - exact fit):")
    print("S =\n", SM1)
    print("Inserted L =\n", result)
    plot_result_s0(SM1, LG1, result,
                   title="Section 0 - Diamond Embedding (5x5 into 9x9)",
                   filename="diamond_embedding_5x9.png")

    print("\n" + "-" * 60)

    # ---- Case 2: 3x4 S into 11x11 L (with centering offset) ----
    SM2 = np.arange(1, 13).reshape(3, 4)
    LG2 = np.zeros((11, 11), dtype=int)
    result2 = embed_diamond(SM2, LG2)
    print("\n[Example 2] S (3x4) into L (11x11):")
    print("S =\n", SM2)
    print("Inserted L =\n", result2)
    plot_result_s0(SM2, LG2, result2,
                   title="Section 0 - Diamond Embedding (3x4 into 11x11)",
                   filename="diamond_embedding_3x11.png")

    print("\n" + "-" * 60)
    # ---- Case 3: Impossible case ----
    SM3 = np.arange(1, 26).reshape(5, 5)
    LG3 = np.zeros((5, 5), dtype=int)
    result3 = embed_diamond(SM3, LG3)
    print(f"\n[Example 3] S (5x5) into L (5x5) -> '{result3}'")

    print("\n[Section 0] All outputs saved to output/utils/")


def test_random_embedding():

    sm_row, sm_col = np.random.randint(2, 7, size=2)
    lg_row, lg_col = np.random.randint(5, 15, size=2)

    LG = np.zeros((lg_row, lg_col), dtype=int)
    SM = np.arange(1, sm_row * sm_col + 1).reshape(sm_row, sm_col)

    result = embed_diamond(SM, LG)

    print(
        f"\n[Random Example] SM ({sm_row}x{sm_col}) into LG ({lg_row}x{lg_col}):")

    if isinstance(result, str) and result == "Impossible":
        print(f"Result: '{result}' (Target matrix is too small)")
    else:
        print("SM =\n", SM)
        print("Inserted LG =\n", result)
        plot_result_s0(SM, LG, result,
                       title=f"Random Diamond Embedding ({sm_row}x{sm_col} into {lg_row}x{lg_col})",
                       filename="random_diamond_embedding.png")


if __name__ == "__main__":

    run_utils()
    test_random_embedding()
