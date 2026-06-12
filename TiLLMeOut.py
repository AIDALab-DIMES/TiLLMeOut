import utils
import numpy as np
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer


class TiLLMeOut:
    '''
    The official implementation of "TiLLMeOut: Time Series Large Language Models-based Outlier Detection".
    '''

    def __init__(self, name, l, u, w, k, it, device):
        '''
        Class Attributes
        - name: name of BART model to use. In the original work "facebook/bart-base" has been used.
        - l: lower bound for the rescaling range. For "facebook/bart-base" l has to be greater than or equal to 0.
        - u: upper bound for the rescaling range. For "facebook/bart-base" u has to be lower than 342.
        - w: size of the window into which the time series is divided (it has to be a mutiple of 4). For "facebook/bart-base" it has to be lower than 1022.
        - it: number of iterations the algoritmh has to perform.
        - k: number of anomalies to consider in each iteration.
        - ids: the input ids corresponding to the numbers into the considere range.
        '''

        self.tokenizer = BartTokenizer.from_pretrained(name)
        self.model = BartForConditionalGeneration.from_pretrained(name)
        self.l = l
        self.u = u
        self.w = w
        self.k = k
        self.it = it
        self.device = device
        self.ids = np.array(self.tokenizer([str(x) for x in range(l, u + 1)]).input_ids)[:, 1]
        self.model.to(device)

    def get_scores(self, timeseries):
        '''
        PARAM:
        timeseries: the univariate time series on which to perform the detection.

        RETURNS:
        scores: a numpy array of reconstruciion error.
        '''

        timeseries_it = utils.normalize_rescale(timeseries, [self.l, self.u])
        error_tk = np.zeros(len(timeseries))

        out = set()

        for _ in tqdm(range(self.it)):

            reconstruction, current_error = self.__reconstruct__(timeseries_it)
            sorted_indexes = current_error.argsort()[-self.k:]

            out.update(sorted_indexes)

            for j in range(self.k):
                if error_tk[sorted_indexes[j]] < current_error[sorted_indexes[j]]:
                    error_tk[sorted_indexes[j]] = current_error[sorted_indexes[j]]

            out_indexes = list(out)
            timeseries_it[out_indexes] = reconstruction[out_indexes]

        scores = current_error
        scores[out_indexes] = error_tk[out_indexes]
        return scores

    def __reconstruct__(self, timeseries):
        '''
        PARAM:
        timeseries: the univariate time series on which to perform the detection.

        RETURNS:
        reconstrucion: the reconstructed time series
        error: a numpy array of reconstrucion error.
        '''

        reconstruction = np.zeros_like(timeseries, dtype=float)

        n = len(timeseries)
        nt = len(self.ids)

        ts_string = np.array([str(x) for x in timeseries], dtype='<U6')

        current_win = [" ".join(ts_string[:self.w])]
        input_ids = self.tokenizer(current_win, return_tensors="pt")["input_ids"].to(self.device)
        values, predictions = self.model(input_ids).logits[0, 1:-1, self.ids].softmax(dim=1).topk(nt)
        values = values.detach().cpu().numpy()[:self.w // 4]
        predictions = predictions.detach().cpu().numpy()[:self.w // 4]
        reconstruction[:self.w // 4] = np.sum(np.multiply(values, predictions), axis=1) + self.l

        for win in range(self.w // 4, n, self.w // 2):
            current_win = [" ".join(ts_string[win - self.w // 4: win + 3 * self.w // 4])]
            input_ids = self.tokenizer(current_win, return_tensors="pt")["input_ids"].to(self.device)
            values, predictions = self.model(input_ids).logits[0, 1:-1, self.ids].softmax(dim=1).topk(nt)
            values = values.detach().cpu().numpy()[self.w // 4: 3 * self.w // 4]
            predictions = predictions.detach().cpu().numpy()[self.w // 4: 3 * self.w // 4]

            reconstruction[win: win + self.w // 2] = np.sum(np.multiply(values, predictions), axis=1) + self.l

        error = timeseries - reconstruction
        error = np.abs((error - np.mean(error)) / np.std(error))

        return reconstruction, error
