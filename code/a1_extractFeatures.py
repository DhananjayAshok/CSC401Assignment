#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
import os

import numpy as np
import argparse
import json
import string
import csv
import warnings

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}


def safe_cast_append(l, value, type=int):
    try:
        l.append(type(value))
    except ValueError:
        pass
    return


def safe_mean(array):
    if len(array) == 0:
        return 0
    else:
        return array.mean()


def safe_std(array):
    if len(array) <= 1:
        return 0
    else:
        return array.std()


def safe_divide(a, b):
    if b != 0:
        return a/b
    else:
        return 0


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros(173)
    # region  Variables
    n_t_upper = 0
    n_fp_pronouns = 0
    n_sp_prononuns = 0
    n_tp_pronouns = 0
    n_coord_conj = 0
    n_pt_verbs = 0
    n_ft_verbs = 0
    n_commas = 0
    n_multi_punc = 0
    n_c_noun = 0
    n_p_noun = 0
    n_adv = 0
    n_wh_word = 0
    n_slang_acro = 0
    avg_l_sent = 0
    avg_len_token = 0
    n_sentences = 0
    # endregion
    # Meta Vars:
    n_tokens = 0
    total_token_length = 0
    total_sentence_length = 0
    going, going_to = False, False
    aoa, img, fam, vmean, amean, dmean = [], [], [], [], [], []
    sentences = comment.split(" \n ")
    n_sentences = len(sentences)
    for sentence in sentences:
        token_tags = sentence.split(" ")
        total_sentence_length += len(token_tags)
        for i, token_tag in enumerate(token_tags):
            if token_tag in ["", " "]:
                continue
            elif "/" not in token_tag:
                #print(f"warning: {token_tag}: {i}th element of sentence \n\t{token_tags} from comment \n\t{sentences} "
                #      f"does not have / and is not an endline")
                #print(f"==\'\': {token_tag==''}, ==\' \': {token_tag==' '}")
                pass
            else:
                split = token_tag.split("/")
                tag = split[-1]
                token = "/".join(split[:-1])
                if len(token) >= 3:
                    if token == token.upper():
                        n_t_upper += 1
                lower = token.lower()
                if lower in FIRST_PERSON_PRONOUNS:
                    n_fp_pronouns += 1
                if lower in SECOND_PERSON_PRONOUNS:
                    n_sp_prononuns += 1
                if lower in THIRD_PERSON_PRONOUNS:
                    n_tp_pronouns += 1
                if tag == "CCONJ":
                    n_coord_conj += 1
                elif tag == "VBD":
                    n_pt_verbs += 1
                if token == ",":
                    n_commas += 1
                elif token in ["will", "gonna"] or "\'ll" in token or (going_to and tag == "VB") :
                    n_ft_verbs += 1
                elif len(token) > 1:
                    flag = True
                    for character in token:
                        if character not in string.punctuation:
                            flag = False
                            break
                    if flag:
                        n_multi_punc += 1
                if token == "going":
                    going = True
                else:
                    if going:
                        if token == "to":
                            going_to = True
                        else:
                            going_to = False
                        going = False

                if tag in ["NN", "NNS"]:
                    n_c_noun += 1
                elif tag in ["NNP", "NNPS"]:
                    n_p_noun += 1
                elif tag == "AD":
                    n_adv += 1
                elif tag in ["WDT", "WP", "WP$", "WRB"]:
                    n_wh_word += 1
                if lower in SLANG:
                    n_slang_acro += 1
                if token not in string.punctuation:
                    n_tokens += 1
                    total_token_length += len(token)
                b_list = b_dict.get(lower, None)
                r_list = r_dict.get(lower, None)
                if b_list is not None:
                    safe_cast_append(aoa, b_list[0])
                    safe_cast_append(img, b_list[1])
                    safe_cast_append(fam, b_list[2])
                if r_list is not None:
                    safe_cast_append(vmean, r_list[0])
                    safe_cast_append(amean, r_list[1])
                    safe_cast_append(dmean, r_list[2])
    avg_l_sent = safe_divide(total_sentence_length, n_sentences)
    avg_len_token = safe_divide(total_token_length, n_tokens)
    feats[0] = n_t_upper
    feats[1] = n_fp_pronouns
    feats[2] = n_sp_prononuns
    feats[3] = n_tp_pronouns
    feats[4] = n_coord_conj
    feats[5] = n_pt_verbs
    feats[6] = n_ft_verbs
    feats[7] = n_commas
    feats[8] = n_multi_punc
    feats[9] = n_c_noun
    feats[10] = n_p_noun
    feats[11] = n_adv
    feats[12] = n_wh_word
    feats[13] = n_slang_acro
    feats[14] = avg_l_sent
    feats[15] = avg_len_token
    feats[16] = n_sentences
    aoa = np.array(aoa)
    img = np.array(img)
    fam = np.array(fam)
    vmean = np.array(vmean)
    amean = np.array(amean)
    dmean = np.array(dmean)
    feats[17] = safe_mean(aoa)
    feats[18] = safe_mean(img)
    feats[19] = safe_mean(fam)
    feats[20] = safe_std(aoa)
    feats[21] = safe_std(img)
    feats[22] = safe_std(fam)
    feats[23] = safe_mean(vmean)
    feats[24] = safe_mean(amean)
    feats[25] = safe_mean(dmean)
    feats[26] = safe_std(vmean)
    feats[27] = safe_std(amean)
    feats[28] = safe_std(dmean)
    return feats


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    index = -1
    for i, id in enumerate(ids[comment_class]):
        if id == comment_id:
            index = i
            break
    if index == -1:
        print(f"Warning, comment {comment_id} of class {comment_class} not found in ids dict")
        index = 0
    feat[29:] = pre_feats[comment_class][index]
    return feat


def main(args):
    #  Declare necessary global variables here.
    cs401_dir = args.a1_dir[:-3]
    b_csv = os.path.join(cs401_dir, "Wordlists", "BristolNorms+GilhoolyLogie.csv")
    b_W = 1
    b_AoA = 3
    b_IMG = 4
    b_FAM = 5
    r_csv = os.path.join(cs401_dir, "Wordlists", "Ratings_Warriner_et_al.csv")
    r_W = 1
    r_V = 2
    r_A = 5
    r_D = 8
    global b_dict, r_dict, cat_dict, pre_feats, ids
    b_dict = {}
    r_dict = {}
    cat_dict = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}
    pre_feats = {}
    ids = {}
    with open(b_csv) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            b_dict[row[b_W]] = [row[b_AoA], row[b_IMG], row[b_FAM]]
    with open(r_csv) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            r_dict[row[r_W]] = [row[r_V], row[r_A], row[r_D]]
    pre_feats["Left"] = np.load(os.path.join(args.a1_dir, "feats", "Left_feats.dat.npy"))
    pre_feats["Right"] = np.load(os.path.join(args.a1_dir, "feats", "Right_feats.dat.npy"))
    pre_feats["Center"] = np.load(os.path.join(args.a1_dir, "feats", "Center_feats.dat.npy"))
    pre_feats["Alt"] = np.load(os.path.join(args.a1_dir, "feats", "Alt_feats.dat.npy"))

    for cat in cat_dict.keys():
        l = []
        with open(os.path.join(args.a1_dir, "feats", f"{cat}_IDs.txt")) as f:
            for i, line in enumerate(f):
                l.append(line.strip())
        ids[cat] = l
    #  Load data
    with open(args.input) as f:
        data = json.load(f)
    feats = np.zeros((len(data), 173+1))
    for i, comment in enumerate(data):
        feats[i, -1] = cat_dict.get(comment["cat"], None)
        c = comment["body"]
        if c in ["", " ", "\n", " \n "]:
            print("Here ", i, " ", c)
        small_feats = extract1(comment["body"])
        feats[i, :-1] = extract2(small_feats, comment["cat"], comment["id"])

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

