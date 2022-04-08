import os
import re
import spacy
import textacy
import logging
import argparse
import json

from comet2.comet_model import PretrainedCometModel
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_TO_QUESTION = {"xIntent": "What was the intention of PersonX?",
                        "xNeed": "Before that, what did PersonX need?",
                        "oEffect": "What happens to others as a result?",
                        "oReact": "What do others feel as a result?",
                        "oWant": "What do others want as a result?",
                        "xEffect": "What happens to PersonX as a result?",
                        "xReact": "What does PersonX feel as a result?",
                        "xWant": "What does PersonX want as a result?",
                        "xAttr": "How is PersonX seen?"}

CATEGORY_TO_PREFIX = {"xIntent": "Because PersonX wanted",
                      "xNeed": "Before, PersonX needed",
                      "oEffect": "Others then",
                      "oReact": "As a result, others feel",
                      "oWant": "As a result, others want",
                      "xEffect": "PersonX then",
                      "xReact": "As a result, PersonX feels",
                      "xWant": "As a result, PersonX wants",
                      "xAttr": "PersonX is seen as"}


def get_clarifications_socialiqa(context, nlp, comet_model):
    """
    Generate clarifications for the SocialIQA dataset
    :param ex: a dictionary with the SocialIQA instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model objects
    :return: a list of (question, answer) tuples
    """
    clarifications = {}
    personx, _ = get_personx(nlp, context)
    # relation = question_to_comet_relation.get(re.sub(personx, "[NAME]", question, flags=re.I), None)
    relu = ["xWant", "xReact", "xAttr", "xIntent", "xNeed", "xEffect"]
    for relation in relu:
        if relation is not None and len(personx) > 0:
            outputs = {relation: comet_model.predict(context, relation, num_beams=1)}
            prefix = CATEGORY_TO_PREFIX[relation]
            for out_event in outputs[relation]:
                if out_event != "none" and out_event != "":
                    if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                        out_event = " ".join((prefix, out_event))
                    out_event = re.sub("personx", str(personx), out_event, flags=re.I)
                    out_event = re.sub("person x", str(personx), out_event, flags=re.I)
                    out_event = re.sub("persony", "others", out_event, flags=re.I)
                    out_event = re.sub("person y", "others", out_event, flags=re.I)
                    # ------------------------------------------------------------------------
                    out_question = CATEGORY_TO_QUESTION[relation]
                    out_question = re.sub("personx", str(personx), out_question, flags=re.I)
                    out_question = re.sub("person x", str(personx), out_question, flags=re.I)
                    out_question = re.sub("persony", "others", out_question, flags=re.I)
                    out_question = re.sub("person y", "others", out_question, flags=re.I)
                    clarifications[out_question] = out_event
    return clarifications


def get_personx(nlp, input_event, use_chunk=True):
    """
    Returns the subject of input_event
    """
    doc = nlp(input_event)
    svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]

    if len(svos) == 0:
        if use_chunk:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
            noun_chunks = [chunk for chunk in doc.noun_chunks]

            if len(noun_chunks) > 0:
                personx = noun_chunks[0].text
                is_named_entity = noun_chunks[0].root.pos_ == "PROP"
            else:
                logger.warning("Didn't find noun chunks either, skipping this sentence.")
                return "", False
        else:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
            return "", False
    else:
        subj_head = svos[0][0]
        # personx = " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])
    try:
        reply = svos[0][0][0], False
    except:
        reply = "", False
    return reply


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=False, default="cpu", help="cpu or GPU device")
    parser.add_argument("--model_file", type=str, required=False, help="The COMET pre-trained model", default=None)
    args = parser.parse_args()

    logger.info(f"Loading COMET model")

    # Load COMET either from args.model_file or from its default location.
    if args.model_file is not None:
        comet_model = PretrainedCometModel(model_name_or_path=args.model_file, device=args.device)
    else:
        comet_model = PretrainedCometModel(device=args.device)

    nlp = spacy.load('en_core_web_sm')

    with open('../../data/dialogue/train.json') as ofile:
        data = json.load(ofile)

    seen = {}
    for key in tqdm(list(data.keys())[62:69]):
        context = data[key]['context']
        # for i in range(0, len(data[key]['turns']), 2):
        #     if context not in seen:
        #         seen[context] = get_clarifications_socialiqa(context, nlp, comet_model)
        #     print(context,seen[context])
        #     context += ' ' + data[key]['turns'][i]
        if context not in seen:
            qna = get_clarifications_socialiqa(context, nlp, comet_model)
            xeta = []
            for x in qna.items():
                xeta += list(x)
            sloppy = '. '.join(xeta)
            context += ' ' + sloppy
            seen[context] = context
        data[key]['clarifications'] = seen[context]
        if int(key) % 1000 == 0:
            with open('../../data/dialogue/train-clarified-2.json',
                    'w+') as ofile:
                json.dump(data, ofile)
                ofile.flush()
                os.fsync(ofile.fileno())
