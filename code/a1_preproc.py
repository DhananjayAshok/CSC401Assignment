#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
 

import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def token_utility(token):
    if token.lemma_ is None or len(token.lemma_) == 0:
        first = token.text
    elif token.lemma_[0] == "-" and (len(token.text) > 0 and token.text[0] !="-"):
        first = token.text
    else:
        if token.text == token.text.upper():
            first = token.lemma_.upper()
        else:
            first = token.lemma_.lower()
    return first + "/" + token.tag_


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  
        # modify this to handle other whitespace chars.
        # replace newlines with spaces
        nonwhitespaces = ["\n", "\t", "\r", "\f", "\v"]
        for item in nonwhitespaces:
            modComm = re.sub(item, " ", modComm)

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"\b(http:\/\/|https:\/\/|www\.)\S+", "", modComm)
        
    if 4 in steps:  # remove duplicate spaces.
        modComm = re.sub(' +', ' ', modComm)

    if 5 in steps:
        # Make sure to: Split tokens with spaces.
        utt = nlp(modComm)
        totalComm = ""
        for i, sent in enumerate(utt.sents):
            if i != 0:
                totalComm += " \n "
            for j, token in enumerate(sent):
                #  print(f"\tLooking at sentence {i} token {j}: {token.text}
                #  has lemma {token.lemma_} and tag {token.tag_}")
                if token.text in [" ", "\t", "\n", "\r", "\f", "\v"]:  # If the token is whitespace ignore but don't add
                    continue
                if j != 0:
                    totalComm += " "
                totalComm += token_utility(token)
        modComm = totalComm
    return modComm


def main(args):
    allOutput = []
    print("Walking through ", indir, " ", list(os.walk(indir)))
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))[:args.max]
            possible_keys = ['author_flair_text', 'parent_id', 'author_flair_css_class', 'link_id', 'ups', 'id',
                             'score_hidden', 'author', 'gilded', 'controversiality', 'name', 'downs', 'archived',
                             'created_utc', 'score', 'distinguished', 'subreddit_id', 'retrieved_on', 'body',
                             'subreddit', 'edited']
            useless_keys = ['parent_id', 'author_flair_css_class', 'link_id',  'subreddit_id', 'author_flair_text',
                            'gilded', 'archived', 'distinguished', 'retrieved_on', 'edited']
            for i, line in enumerate(data):
                j = json.loads(line)
                keys = list(j.keys())
                for key in useless_keys:
                    if key in keys and key != "body":
                        j.pop(key)
                j["cat"] = file
                processed_body = preproc1(j["body"])
                if processed_body == "":
                    # print(f"{j['body']} is empty when processed")
                    pass
                else:
                    j["body"] = processed_body
                    allOutput.append(j)
    print(f"allOutput length: {len(allOutput)}")
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each.')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID', default=1004842977)
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. "
                                         "Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if args.max > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    print(args)
    main(args)
