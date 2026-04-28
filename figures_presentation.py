import matplotlib.pyplot as plt
import figures_article as fa


def _clear_figure_legends(fig, debug=False):
    n = len(fig.legends)
    for lg in list(fig.legends):
        lg.remove()
    if debug:
        print(f"_clear_figure_legends output: removed={n}")


def _add_inside_legend(ax, handles, labels, loc="upper left", ncol=1, fontsize=9, debug=False):
    if not handles:
        return
    # Keep label order while removing duplicates.
    dedup = dict(zip(labels, handles))
    legend = ax.legend(
        dedup.values(),
        dedup.keys(),
        loc=loc,
        ncol=1,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
    )
    # Keep legend as overlay; don't let layout engine resize panels for it.
    legend.set_in_layout(False)
    if debug:
        print(f"_add_inside_legend output: n_labels={len(dedup)}")


def _save_presentation(fig, output_path, width=14.0, height=8.0, debug=False):
    fig.set_size_inches(width, height, forward=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    if debug:
        print(f"_save_presentation output: file={output_path}, size=({width}, {height})")


def main(debug=False):
    if debug:
        print("main input: debug=True")

    fig4 = fa.figure_4_carbon(savefig=False, debug=debug)
    _clear_figure_legends(fig4, debug=debug)
    if len(fig4.axes) >= 4:
        h_top, l_top = fig4.axes[0].get_legend_handles_labels()
        h_bot, l_bot = fig4.axes[2].get_legend_handles_labels()
        _add_inside_legend(fig4.axes[0], h_top, l_top, loc="upper left", ncol=1, fontsize=10, debug=debug)
        _add_inside_legend(fig4.axes[2], h_bot, l_bot, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig4, "results_figures/figure_4_carbon_presentation.png", width=10, height=8.6, debug=debug)

    fig5 = fa.figure_5_prices(savefig=False, debug=debug, PHASEOUT=False)
    _clear_figure_legends(fig5, debug=debug)
    if len(fig5.axes) >= 6:
        h_top, l_top = fig5.axes[0].get_legend_handles_labels()
        h_bot, l_bot = fig5.axes[3].get_legend_handles_labels()
        _add_inside_legend(fig5.axes[0], h_top, l_top, loc="upper left", ncol=1, fontsize=10, debug=debug)
        _add_inside_legend(fig5.axes[3], h_bot, l_bot, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig5, "results_figures/figure_5_prices_presentation.png", width=10, height=8.6, debug=debug)

    fig6 = fa.figure_6_policyregret(savefig=False, debug=debug)
    _clear_figure_legends(fig6, debug=debug)
    if len(fig6.axes) >= 2:
        h, l = fig6.axes[0].get_legend_handles_labels()
        _add_inside_legend(fig6.axes[0], h, l, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig6, "results_figures/figure_6_policyregret_presentation.png", width=10, height=8.6, debug=debug)

    fig7 = fa.figure_7_gasregret(savefig=False, debug=debug)
    _clear_figure_legends(fig7, debug=debug)
    if len(fig7.axes) >= 2:
        h, l = fig7.axes[0].get_legend_handles_labels()
        _add_inside_legend(fig7.axes[0], h, l, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig7, "results_figures/figure_7_gasregret_presentation.png", width=10, height=8.6, debug=debug)

    plt.show()


if __name__ == "__main__":
    main(debug=True)
