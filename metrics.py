import Levenshtein as Lev

def cer(ref, hyp):
    # Character Error Rate
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    return Lev.distance(ref, hyp) / max(1, len(ref))


def wer(ref, hyp):
    # Word error rate
    ref_words = ref.split()
    hyp_words = hyp.split()
    return Lev.distance(" ".join(ref_words), " ".join(hyp_words)) / max(1, len(ref_words))