class CorrMatrix:
    def __init__(self):
        pass

    @staticmethod
    def get_corr_matrix(real_df, syn_df, nums):
        return {"real": round(real_df[nums].corr(), 3),
                "syn": round(syn_df[nums].corr(), 3),
                "diff": round(real_df[nums].corr() - syn_df[nums].corr(), 3)}
