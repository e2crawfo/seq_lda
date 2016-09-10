

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument(
        '--quick', action='store_true',
        help='If True, will run fresh experiments '
             'only there are no existing results. ')
    args = parser.parse_args()

    Plot = namedtuple('plot', 'title measure_name x_var_name x_var_values '
                              'measure_display x_var_display kwargs')

    kwargs = dict(
        alpha=0.01,
        beta=1.0,
        length=np.inf,
        n_symbols=10,
        n_train=50,
        n_test=10,
        n_topics=10,
        n_words_per_doc=100)

    if args.quick:
        plots = dict(
            vs_n_topics=Plot(
                title='Performance on Test Set',
                measure_name='test_likelihood_lower_bound',
                x_var_name='n_topics',
                x_var_values=[2, 4],
                measure_display='Log Likelihood Lower Bound',
                x_var_display='\# Topics',
                kwargs=dict()),
            vs_n_train=Plot(
                title='Performance on Test Set',
                measure_name='test_likelihood_lower_bound',
                x_var_name='n_train',
                x_var_values=[2, 4],
                measure_display='Log Likelihood Lower Bound',
                x_var_display='\# Training Documents',
                kwargs=dict()),
            vs_n_words_per_doc=Plot(
                title='Performance on Test Set',
                measure_name='test_likelihood_lower_bound',
                x_var_name='n_words_per_doc',
                x_var_values=[10, 20],
                measure_display='Log Likelihood Lower Bound',
                x_var_display='\# Words per Doc',
                kwargs=dict()),
            )
    else:
        plots = dict(
                vs_n_topics=Plot(
                    title='Performance on Test Set',
                    measure_name='test_likelihood_lower_bound',
                    x_var_name='n_topics',
                    x_var_values=np.arange(2, 10),
                    measure_display='Log Likelihood Lower Bound',
                    x_var_display='\# Topics',
                    kwargs=dict()),
                vs_n_train=Plot(
                    title='Performance on Test Set',
                    measure_name='test_likelihood_lower_bound',
                    x_var_name='n_train',
                    x_var_values=np.arange(2, 20, 2),
                    measure_display='Log Likelihood Lower Bound',
                    x_var_display='\# Training Documents',
                    kwargs=dict()),
                vs_n_words_per_doc=Plot(
                    title='Performance on Test Set',
                    measure_name='test_likelihood_lower_bound',
                    x_var_name='n_words_per_doc',
                    x_var_values=np.arange(10, 110, 10),
                    measure_display='Log Likelihood Lower Bound',
                    x_var_display='\# Words per Doc',
                    kwargs=dict()),
                )

    func = run_markov_medium

    plot_names = 'vs_n_train vs_n_words_per_doc vs_average_word_length vs_n_topics'
    for plot_name in plot_names.split(' '):
        print("Running experiment %s." % plot_name)

        _kwargs = kwargs.copy()

        plot = plots[plot_name]
        title = plot.title
        measure_name = plot.measure_name
        x_var_name = plot.x_var_name
        x_var_values = plot.x_var_values
        measure_display = plot.measure_display
        x_var_display = plot.x_var_display
        _kwargs.update(plot.kwargs)

        results_filename = os.path.join(
            'quick_data' if args.quick else 'data', plot_name + '.csv')
        plot_filename = os.path.join(
            'quick_plots' if args.quick else 'plots', plot_name + '.pdf')

        if not args.plot or not os.path.exists(results_filename):
            df = run_experiment(
                measure_name, x_var_name, x_var_values,
                func, **_kwargs)

            df.to_csv(results_filename)
        else:
            df = pd.read_csv(results_filename)

        markers = MarkerStyle.filled_markers
        colors = seaborn.xkcd_rgb.values()

        labels = dict(ground_truth='Ground Truth',
                      markov='LDA with\nMarkov Topics',
                      lda='LDA')

        def plot_kwarg_func(sv):
            idx = hash(str(sv))
            marker = markers[idx % len(markers)]
            c = colors[idx % len(colors)]
            label = labels[sv]
            return dict(label=label, c=c, marker=marker)

        plt.figure()
        plot_measures(
            df, measure_name, x_var_name, 'method',
            legend_outside=False,
            kwarg_func=plot_kwarg_func,
            measure_display=measure_display,
            x_var_display=x_var_display)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=22)

        plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.12)
        plt.title(title)

        plt.gcf().set_size_inches(10, 5, forward=True)
        p
