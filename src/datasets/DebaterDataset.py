from torch.utils.data.dataset import Dataset
import pandas as pd

#   Dataset class for the combined classifier

class DebaterDataset(Dataset):
    def __init__(self, path, is_test):
        data = pd.read_csv(path)
        self.label_list = data.loc[:, 'label'].tolist()
        self.body_parent_list = data.loc[:, 'body_parent'].tolist()
        self.body_child_list = data.loc[:, 'body_child'].tolist()
        self.submission_text_list = data.loc[:, 'submission_text'].tolist()

        L = len(self.label_list)
        p = 1.0
        if not is_test:
            self.label_list = self.label_list[:int(L * 0.8 * p)]
            self.body_parent_list = self.body_parent_list[:int(L * 0.8 * p)]
            self.body_child_list = self.body_child_list[:int(L * 0.8 * p)]
            self.submission_text_list = self.submission_text_list[:int(L * 0.8 * p)]
        else:
            self.label_list = self.label_list[-int(L * 0.2 * p):]
            self.body_parent_list = self.body_parent_list[-int(L * 0.2 * p):]
            self.body_child_list = self.body_child_list[-int(L * 0.2 * p):]
            self.submission_text_list = self.submission_text_list[-int(L * 0.2 * p):]

    def __len__(self):
        return self.label_list.__len__()

    def __getitem__(self, index):
        '''
          The inidividual item returned by __getitem__ should be a dictionary
          containing the following keys:
          - parent_comment: string
          - child_comment: string
          - context: string
          - label: int (0: Neutral, 1: Agree, 2: Disagree)
        '''
        label = self.label_list[index]
        parent_comment = self.body_parent_list[index]
        child_comment = self.body_child_list[index]
        context = self.submission_text_list[index]

        return {'parent_comment': parent_comment, \
                'child_comment': child_comment, \
                'context': context, \
                'label': label}