"""Render CUDA kernels with explicit memory layouts and head dispatch tables."""

from pathlib import Path
from string import Template

from visionhope.utils.cuda import CUDA_GROUP_SOURCE

_SOURCE_DIR = Path(__file__).parent / "csrc"


def _render(name, **parameters):
    return Template((_SOURCE_DIR / name).read_text()).substitute(parameters)


def memory_helpers():
    return _render("memory.cuh", thread_groups=CUDA_GROUP_SOURCE)


def _chunk_dispatch(macro, *, head64=False, raw=False):
    kind = "qraw" if raw else "CUDA"
    if head64:
        return (f"    if (D_int == 64) {{ {macro}(64) }}\n"
                f'    else {{ throw std::runtime_error("HOPE D64 {kind} kernel solely supports head_dim=64."); }}')
    return (f"    if (D_int == 4)       {{ {macro}(4) }}\n"
            f"    else if (D_int == 8)  {{ {macro}(8) }}\n"
            f"    else if (D_int == 16) {{ {macro}(16) }}\n"
            f"    else if (D_int == 32) {{ {macro}(32) }}\n"
            f'    else {{ throw std::runtime_error("HOPE {kind} kernel solely supports head_dim in {{4, 8, 16, 32}}."); }}')


def chunk_source(*, head64=False):
    backward_raw = (
        '    if (D_int == 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(64, 1, __D64_CMAX__) }\n'
        '    else { throw std::runtime_error("HOPE D64 qraw kernel solely supports head_dim=64."); }'
        if head64 else (_SOURCE_DIR / "chunk_backward_dispatch.cuh").read_text().rstrip("\n")
    )
    return _render(
        "chunk.cu",
        forward_dispatch=_chunk_dispatch("DISPATCH_SRNL_FWD", head64=head64),
        forward_raw_dispatch=_chunk_dispatch("DISPATCH_SRNL_FWD_QRAW", head64=head64, raw=True),
        backward_dispatch=_chunk_dispatch("DISPATCH_SRNL_BWD", head64=head64),
        backward_raw_dispatch=backward_raw,
    )


_LINE_LAYOUTS = {
    "large": {
        "direction_check": '(direction_count != 1 && direction_count != 2 && direction_count != 4)',
        "kernel_parameters": (
            'int N, int L, int ChunkSize, int NumChunks, int StateCount,\n'
            '    int NumHeads, int SaveQInv, int DirectionCount, int MaxSubchunks)'
        ),
        "state_layout": (
            'int q_channels = NumHeads * D;\n'
            '\n'
        ),
        "prefix_index": 'pid * (NumChunks - 1) * D * D + (c - 1) * D * D',
        "checkpoint_write_index": 'pid * (NumChunks * MaxSubchunks) * D * D + save_idx * D * D',
        "checkpoint_read_index": 'pid * (NumChunks * MaxSubchunks) * D * D + (c * MaxSubchunks + sub) * D * D',
        "final_index": 'pid * NumChunks * D * D + c * D * D',
        "call_parameters": 'N, L, chunk_size, num_chunks, state_count, num_heads, save_qinv, direction_count, max_subchunks);',
        "api_parameters": 'int chunk_size, int D_int, int state_count, int num_heads, int save_qinv, int max_subchunks) {',
        "chunk_counts": (
            'int num_chunks = L / chunk_size;\n'
            '    int direction_count = QRaw.size(1);'
        ),
        "parallel_length": 'chunk_size',
    },
    "varline": {
        "direction_check": 'direction_count != 4',
        "kernel_parameters": (
            'int N, int L, int RowChunkSize, int ColChunkSize, int RowNumChunks, int ColNumChunks, int MaxNumChunks, int StateCount,\n'
            '    int NumHeads, int SaveQInv, int DirectionCount, int MaxSubchunks)'
        ),
        "state_layout": (
            'int q_channels = NumHeads * D;\n'
            '    int ChunkSize = (q_dir < 2) ? RowChunkSize : ColChunkSize;\n'
            '    int NumChunks = (q_dir < 2) ? RowNumChunks : ColNumChunks;\n'
            '\n'
            '    int bh = q_b * NumHeads + q_head;\n'
            '    int row_prefix_chunks = RowNumChunks > 0 ? RowNumChunks - 1 : 0;\n'
            '    int col_prefix_chunks = ColNumChunks > 0 ? ColNumChunks - 1 : 0;\n'
            '    int chunks_per_bh = 2 * RowNumChunks + 2 * ColNumChunks;\n'
            '    int prefix_chunks_per_bh = 2 * row_prefix_chunks + 2 * col_prefix_chunks;\n'
            '    int chunk_base = bh * chunks_per_bh + ((q_dir < 2) ? q_dir * RowNumChunks : 2 * RowNumChunks + (q_dir - 2) * ColNumChunks);\n'
            '    int prefix_base = bh * prefix_chunks_per_bh + ((q_dir < 2) ? q_dir * row_prefix_chunks : 2 * row_prefix_chunks + (q_dir - 2) * col_prefix_chunks);\n'
            '\n'
        ),
        "prefix_index": '(prefix_base + (c - 1)) * D * D',
        "checkpoint_write_index": '(chunk_base * MaxSubchunks + save_idx) * D * D',
        "checkpoint_read_index": '(chunk_base * MaxSubchunks + c * MaxSubchunks + sub) * D * D',
        "final_index": '(chunk_base + c) * D * D',
        "call_parameters": 'N, L, row_chunk_size, col_chunk_size, row_num_chunks, col_num_chunks, max_num_chunks, state_count, num_heads, save_qinv, direction_count, max_subchunks);',
        "api_parameters": 'int row_chunk_size, int col_chunk_size, int D_int, int state_count, int num_heads, int save_qinv, int max_num_chunks, int max_subchunks) {',
        "chunk_counts": (
            'int row_num_chunks = L / row_chunk_size;\n'
            '    int col_num_chunks = L / col_chunk_size;\n'
            '    int direction_count = QRaw.size(1);'
        ),
        "parallel_length": 'std::max(row_chunk_size, col_chunk_size)',
    },
}


def _line_dispatch(layout, *, head64=False, backward=False):
    operation = "BWD" if backward else "FWD"
    macro = f"DISPATCH_{layout.upper()}_{operation}"
    dimensions = (64,) if head64 else (4, 8, 16, 32)
    branches = []
    for index, dimension in enumerate(dimensions):
        condition = "if" if index == 0 else "else if"
        arguments = f"{dimension}, {max(1, 32 // dimension)}" if backward else str(dimension)
        branches.append(f"    {condition} (D_int == {dimension}) {{ {macro}({arguments}) }}")
    if head64:
        kind = "varline" if layout == "varline" else "large-line"
        suffix = "backward " if backward else ""
        message = f"{kind} qraw D64 {suffix}extension solely supports head_dim=64."
    else:
        suffix = "backward " if backward else ""
        message = f"large-line qraw {suffix}supports head_dim in {{4, 8, 16, 32}}."
    branches.append(f'    else {{ throw std::runtime_error("{message}"); }}')
    return "\n".join(branches)


def line_source(*, variable=False, head64=False):
    layout = "varline" if variable else "large"
    source = _render(
        "lines.cu", layout=layout, macro_layout=layout.upper(),
        forward_dispatch=_line_dispatch(layout, head64=head64),
        backward_dispatch=_line_dispatch(layout, head64=head64, backward=True),
        **_LINE_LAYOUTS[layout],
    )
    return source.replace("#pragma unroll", "#pragma unroll 1") if head64 else source
