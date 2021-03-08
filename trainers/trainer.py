from tqdm import trange
class Trainer(object):
    def __init__(self,params,model,dataset_train):
        super().__init__()

        self._model = model
        self._dataset_train = dataset_train


        #steps
        self._max_steps = int(params.max_steps)
        self._log_every_n_steps = int(params.log_every_n_steps)
        self._save_every_n_steps = int(params.save_every_n_steps)

        #dataset
        self._batch_size = params.batch_size


        # optimizer
        self._loss_fn = params.loss_fn
        self._optimizer_cls = params.optimizer_cls
        self._learning_rate = params.learning_rate


        #TODO:
        #create optimizer and checkpoint
        #tensorboard logging
        #session and init
        #time

    def _setup_model(self,dataset,is_train,summary_prefix=""):
        #TODO:
        pass
    
    def run(self):
        #TODO:
        #restore checkpoint
        #run
        self._model.train()
        t = trange(10)
        for i in t:
            # fetch the next training batch
            train_batch,labels_batch = next(self.data_iterator)
            # compute model output and loss
            output_batch = self._model(train_batch)
            loss = self._loss_fn(output_batch,labels_batch)

            #clear pervious gradients
            self._optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # perfoms updates
            self._optimizer.step()

            if i % self._save_every_n_steps:
                self._save()
        #save
        #log
        pass
    def _get_global_step(self):
        #TODO
        pass
    def _train_step(self):
        #TODO
        pass
    def _log(self):
        #TODO
        pass
    def _save(self):
        pass
