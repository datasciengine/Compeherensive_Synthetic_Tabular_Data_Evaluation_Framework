class CatFreqCalculator:
    def __init__(self):
        pass

    @staticmethod
    def get_cat_freq_calculator(real_df, syn_df, cats):
        cat_freqs = {}
        for cat in cats:
            real_counts = round(real_df[cat].value_counts(normalize=True), 3)
            syn_counts = round(syn_df[cat].value_counts(normalize=True), 3)

            cat_freqs[cat] = {"real": real_counts,
                              "syn": syn_counts,
                              "diff": round(real_counts - syn_counts, 3)}
        return cat_freqs
