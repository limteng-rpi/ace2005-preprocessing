from xml.etree import ElementTree
from bs4 import BeautifulSoup
import nltk
import json
import re


class Parser:
    def __init__(self, path):
        self.path = path
        self.entity_mentions = []
        self.event_mentions = []
        self.sentences = []
        self.sgm_text = ''

        (
            self.entity_mentions,
            self.event_mentions,
            self.relation_mentions,
            self.doc_id
        ) = self.parse_xml(path + '.apf.xml')
        self.sents_with_pos, self.headline = self.parse_sgm(path + '.sgm')
        self.fix_wrong_position()

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ')

    def get_data(self):
        data = []
        for sent_idx, sent in enumerate(self.sents_with_pos):
            item = dict()

            item['sentence'] = self.clean_text(sent['text'])
            item['position'] = sent['position']
            item['headline'] = sent_idx == 0 and self.headline is not None and self.headline.text.strip() == item['sentence'].strip()
            # print(self.doc_id, self.headline.text.strip(), item['sentence'])
            text_position = sent['position']


            for i, s in enumerate(item['sentence']):
                if s != ' ':
                    item['position'][0] += i
                    break

            item['sentence'] = item['sentence'].strip()

            entity_map = dict()
            item['golden_entity_mentions'] = []
            item['golden_event_mentions'] = []
            item['golden_relation_mentions'] = []
            item['doc_id'] = self.doc_id

            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']
                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:
                    item['golden_entity_mentions'].append({
                        'id': entity_mention['entity_id'],
                        'text': self.clean_text(entity_mention['text']),
                        'position': entity_position,
                        'entity_type': entity_mention['entity_type'],
                        'mention_type': entity_mention['mention_type'],
                        'head_text': entity_mention['head_text'],
                        'head_position': entity_mention['head_position'],
                    })
                    entity_map[entity_mention['entity_id']] = entity_mention

            for event_mention in self.event_mentions:
                event_position = event_mention['trigger']['position']
                if text_position[0] <= event_position[0] and event_position[1] <= text_position[1]:
                    event_arguments = []
                    for argument in event_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity_id']]['entity_type']
                        except KeyError:
                            print('[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue

                        event_arguments.append({
                            'role': argument['role'],
                            'position': argument['position'],
                            'entity_type': entity_type,
                            'text': self.clean_text(argument['text']),
                        })

                    item['golden_event_mentions'].append({
                        'id': event_mention['event_id'],
                        'trigger': event_mention['trigger'],
                        'arguments': event_arguments,
                        'position': event_position,
                        'event_type': event_mention['event_type'],
                    })

            for relation_mention in self.relation_mentions:
                relation_position = relation_mention['position']
                if text_position[0] <= relation_position[0] and relation_position[1] <= text_position[1]:
                    relation_arguments = []
                    for argument in relation_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity_id']]['entity_type']
                        except KeyError:
                            print(
                                '[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue

                        relation_arguments.append({
                            'role': argument['role'],
                            'position': argument['position'],
                            'entity_type': entity_type,
                            'text': self.clean_text(argument['text']),
                        })

                    item['golden_relation_mentions'].append({
                        'arguments': relation_arguments,
                        'position': relation_position,
                        'relation_type': relation_mention['relation_type'],
                    })

            data.append(item)
        return data

    def find_correct_offset(self, sgm_text, start_index, text):
        offset = 0
        for i in range(0, 70):
            for j in [-1, 1]:
                offset = i * j
                if sgm_text[start_index + offset:start_index + offset + len(text)] == text:
                    return offset

        print('[Warning] fail to find offset! (start_index: {}, text: {}, path: {})'.format(start_index, text, self.path))
        return offset

    def fix_wrong_position(self):
        for entity_mention in self.entity_mentions:
            offset = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=entity_mention['position'][0],
                text=entity_mention['text'])

            entity_mention['position'][0] += offset
            entity_mention['position'][1] += offset
            entity_mention['head_position'][0] += offset
            entity_mention['head_position'][1] += offset

        for event_mention in self.event_mentions:
            offset1 = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=event_mention['trigger']['position'][0],
                text=event_mention['trigger']['text'])
            event_mention['trigger']['position'][0] += offset1
            event_mention['trigger']['position'][1] += offset1

            for argument in event_mention['arguments']:
                offset2 = self.find_correct_offset(
                    sgm_text=self.sgm_text,
                    start_index=argument['position'][0],
                    text=argument['text'])
                argument['position'][0] += offset2
                argument['position'][1] += offset2

        for relation_mention in self.relation_mentions:
            for argument in relation_mention['arguments']:
                offset2 = self.find_correct_offset(
                    sgm_text=self.sgm_text,
                    start_index=argument['position'][0],
                    text=argument['text'])
                argument['position'][0] += offset2
                argument['position'][1] += offset2

    def parse_sgm(self, sgm_path):
        with open(sgm_path, 'r') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            self.sgm_text = soup.text

            doc_type = soup.doc.doctype.text.strip()

            def remove_tags(selector):
                tags = soup.findAll(selector)
                for tag in tags:
                    tag.extract()

            if doc_type == 'WEB TEXT':
                remove_tags('poster')
                remove_tags('postdate')
                remove_tags('subject')
            elif doc_type in ['CONVERSATION', 'STORY']:
                remove_tags('speaker')

            sents = []
            converted_text = soup.text

            for sent in nltk.sent_tokenize(converted_text):
                sents.extend(sent.split('\n\n'))
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = sents[1:]
            sents_with_pos = []
            last_pos = 0
            for sent in sents:
                pos = self.sgm_text.find(sent, last_pos)
                last_pos = pos
                sents_with_pos.append({
                    'text': sent,
                    'position': [pos, pos + len(sent)]
                })

            return sents_with_pos, soup.doc.body.headline

    def parse_xml(self, xml_path):
        entity_mentions, event_mentions, relation_mentions = [], [], []
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()
        doc_id = root[0].attrib['DOCID']

        for child in root[0]:
            if child.tag == 'entity':
                entity_mentions.extend(self.parse_entity_tag(child))
            elif child.tag in ['value', 'timex2']:
                entity_mentions.extend(self.parse_value_timex_tag(child))
            elif child.tag == 'event':
                event_mentions.extend(self.parse_event_tag(child))
            elif child.tag == 'relation':
                relation_mentions.extend(self.parse_relation_tag(child))

        return entity_mentions, event_mentions, relation_mentions, doc_id

    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag != 'entity_mention':
                continue
            extent = child[0]
            charseq = extent[0]
            head = child[1]
            head_charseq = head[0]
            assert extent.tag == 'extent' and head.tag == 'head'

            entity_mention = dict()
            entity_mention['entity_id'] = child.attrib['ID']
            entity_mention['entity_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
            entity_mention['mention_type'] = '{}:{}'.format(child.attrib['TYPE'], child.attrib['LDCTYPE'])
            entity_mention['text'] = charseq.text
            entity_mention['position'] = [int(charseq.attrib['START']), int(charseq.attrib['END'])]
            entity_mention['head_text'] = head_charseq.text
            entity_mention['head_position'] = [int(head_charseq.attrib['START']),
                                               int(head_charseq.attrib['END'])]

            entity_mentions.append(entity_mention)

        return entity_mentions

    @staticmethod
    def parse_event_tag(node):
        event_mentions = []
        for child in node:
            if child.tag == 'event_mention':
                event_mention = dict()
                event_mention['event_id'] = child.attrib['ID']
                event_mention['event_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                event_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'ldc_scope':
                        charseq = child2[0]
                        event_mention['text'] = charseq.text
                        event_mention['position'] = [int(charseq.attrib['START']), int(charseq.attrib['END'])]
                    if child2.tag == 'anchor':
                        charseq = child2[0]
                        event_mention['trigger'] = {
                            'text': charseq.text,
                            'position': [int(charseq.attrib['START']), int(charseq.attrib['END'])],
                        }
                    if child2.tag == 'event_mention_argument':
                        extent = child2[0]
                        charseq = extent[0]
                        event_mention['arguments'].append({
                            'text': charseq.text,
                            'position': [int(charseq.attrib['START']), int(charseq.attrib['END'])],
                            'role': child2.attrib['ROLE'],
                            'entity_id': child2.attrib['REFID'],
                        })
                event_mentions.append(event_mention)
        return event_mentions

    @staticmethod
    def parse_relation_tag(node):
        relation_mentions = []
        for child in node:
            if child.tag == 'relation_mention':
                relation_mention = dict()
                relation_mention['relation_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                relation_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'extent':
                        charseq = child2[0]
                        relation_mention['text'] = charseq.text
                        relation_mention['position'] = [int(charseq.attrib['START']),
                                                        int(charseq.attrib['END'])]
                    if child2.tag == 'relation_mention_argument':
                        extent = child2[0]
                        charseq = extent[0]
                        relation_mention['arguments'].append({
                            'text': charseq.text,
                            'position': [int(charseq.attrib['START']),
                                         int(charseq.attrib['END'])],
                            'role': child2.attrib['ROLE'],
                            'entity_id': child2.attrib['REFID']
                        })
                relation_mentions.append(relation_mention)
        return relation_mentions

    @staticmethod
    def parse_value_timex_tag(node):
        entity_mentions = []

        for child in node:
            extent = child[0]
            charseq = extent[0]

            entity_mention = dict()
            entity_mention['entity_id'] = child.attrib['ID']

            if 'TYPE' in node.attrib:
                entity_mention['entity_type'] = node.attrib['TYPE']
            if 'SUBTYPE' in node.attrib:
                entity_mention['entity_type'] += ':{}'.format(node.attrib['SUBTYPE'])
            if child.tag == 'timex2_mention':
                entity_mention['entity_type'] = 'TIM:time'
                entity_mention['mention_type'] = 'TIM:TIM'
            elif child.tag == 'value_mention':
                entity_mention['mention_type'] = 'VAL:VAL'

            entity_mention['text'] = charseq.text
            entity_mention['position'] = [int(charseq.attrib['START']), int(charseq.attrib['END'])]
            entity_mention['head_text'] = charseq.text
            entity_mention['head_position'] = [int(charseq.attrib['START']), int(charseq.attrib['END'])]

            entity_mentions.append(entity_mention)

        return entity_mentions


if __name__ == '__main__':
    # parser = Parser('./data/ace_2005_td_v7/data/English/un/fp2/alt.gossip.celebrities_20041118.2331')
    parser = Parser('./data/ace_2005_td_v7/data/English/un/timex2norm/alt.corel_20041228.0503')
    data = parser.get_data()
    with open('./output/debug.json', 'w') as f:
        json.dump(data, f, indent=2)

    # index = parser.sgm_text.find("Diego Garcia")
    # print('index :', index)
    # print(parser.sgm_text[1918 - 30:])
