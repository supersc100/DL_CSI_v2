function plot_sampling_overhead()
%PLOT_SAMPLING_OVERHEAD  Plot sampling-strategy ablation vs overhead.
%
%   Reads outputs/sim/results_sampling_overhead.csv.

    csvPath = fullfile('..', '..', 'outputs', 'sim', 'results_sampling_overhead.csv');
    outDir  = fullfile('..', '..', 'outputs', 'sim');

    if ~isfile(csvPath)
        error('CSV not found: %s', csvPath);
    end

    T = readtable(csvPath);

    lines = {
        'uniform',    [0.000 0.447 0.741], '-',  'o', 'Uniform';
        'nonuniform', [0.494 0.184 0.556], '--', 's', 'Non-uniform';
        'adaptive',   [0.850 0.325 0.098], '-.', '^', 'Adaptive';
    };

    fig = figure('Name', 'Sampling Strategy vs Overhead', 'Color', 'w');
    hold on; box on; grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1);

    hLeg = [];
    legLabels = {};
    for i = 1:size(lines, 1)
        name = lines{i, 1};
        idx = strcmp(T.curve, name);
        if ~any(idx), continue; end

        xVals = T.x(idx);
        yVals = T.mean(idx);
        [xSort, order] = sort(xVals);
        ySort = yVals(order);

        h = plot(xSort, ySort, ...
            'Color',     lines{i, 2}, ...
            'LineStyle', lines{i, 3}, ...
            'Marker',    lines{i, 4}, ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6, ...
            'MarkerFaceColor', lines{i, 2});
        hLeg(end+1) = h;
        legLabels{end+1} = lines{i, 5};
    end

    xlabel('Pilot overhead (%)', 'FontSize', 13);
    ylabel('NMSE (dB)', 'FontSize', 13);
    title('Sampling strategy: NMSE versus overhead', 'FontSize', 14);
    legend(hLeg, legLabels, 'Location', 'best', 'FontSize', 10);

    if ~isfolder(outDir), mkdir(outDir); end
    savefig(fig, fullfile(outDir, 'fig_sampling_overhead_matlab.fig'));
    print(fig, fullfile(outDir, 'fig_sampling_overhead_matlab.png'), '-dpng', '-r300');

    fprintf('Saved:\n  %s\n  %s\n', ...
        fullfile(outDir, 'fig_sampling_overhead_matlab.fig'), ...
        fullfile(outDir, 'fig_sampling_overhead_matlab.png'));
end
