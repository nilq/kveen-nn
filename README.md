<h1 align="center">
  <img src="https://i.ibb.co/qpGDnQQ/dronning-rnn.png" width="20%">
</h1>
<h1 align="center">Margrethe 2.0</h1>

## Queen LSTM char-based RNN

This project contains a spider, that automatically crawls the royal family's website for links to previous new year's eve speeches. The spider gathers everything in `speeches.dat`. Once the data is there, a simple LSTM recurrent neural network auto-encoder can start training on it.

### Making speeches

`pip install -r requirenments`
`python kween.py`

The `kween.py` can be run with `--gpu` to use CUDA to run on the GPU. It will auto-save the RNN model on every 500 epochs. You can continue training where you left of by using `--based <your saved model.pt>`. The two flags can of course be used together.

## Example Speeches

My computer isn't super strong, and I haven't gotten around to properly setting up ROCM, thus I've been forced to train on my CPU.

With an average loss of `0.5092` the speeches look something like this:

> \[120m 24s (epoch 2848 : 36%) @ loss 0.4798\]
> Vi skal leve os om professionelle resten og jeg har min familiemedlem. Men det er vi stadig omkring, ikke ligger for vores bade tidligere og inspirer. Det var det dansk.  Vi skal vaere sammen til alle i Gronland og de unge bonde i Danmark, som kun familien. Til de matte ogsa sig et godt nytar for at de er nytarsaften - men hele verden. Jeg sender mine gode onsker. Det kan side med at blive vise dem. I farten for alle i verden os videre: Det skal man har modt plivis og ansvarsbemmillitet.

Very cool. Thank you queen.

## License

Do whatever.
