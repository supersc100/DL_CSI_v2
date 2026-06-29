function plot_nmse_overhead()
%PLOT_NMSE_OVERHEAD  Plot Phase 2 NMSE vs pilot overhead from run_simulation.py.
%
%   Reads outputs/sim/results_nmse_overhead.csv.  Duplicate x entries are
%   averaged so the plot is clean.

    csvPath = fullfile('..', '..', 'outputs', 'sim', 'results_nmse_overhead.csv');
    outDir  = fullfile('..', '..', 'outputs', 'sim');

    if ~isfile(csvPath)
        error('CSV not found: %s', csvPath);
    end

    T = readtable(csvPath);

    % Some x values are duplicated in the CSV; average them per curve/x.
    [G, curveId, xId] = findgroups(T.curve, T.x);
    yMean = splitapply(@mean, T.mean, G);
    yStd  = splitapply(@mean, T.std,  G);
    U = table(curveId, xId, yMean, yStd, ...
              'VariableNames', {'curve', 'x', 'mean', 'std'});

    lines = {
        'proposed',      [0.000 0.447 0.741], '-',  'o', 'Proposed';
        'linear_interp', [0.494 0.184 0.556], '--', '>', 'Linear interp';
        'dft_interp',    [0.850 0.325 0.098], '-.', '<', 'DFT interp';
        'magnitude_only',[0.466 0.674 0.188], ':',  'x', 'Magnitude only';
        'full_feedback', [0.500 0.500 0.500], ':',  'none', 'Full feedback';
    };

    fig = figure('Name', 'Phase 2 NMSE vs Overhead', 'Color', 'w');
    hold on; box on; grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1);

    hLeg = [];
    legLabels = {};
    for i = 1:size(lines, 1)
        name = lines{i, 1};
        idx = strcmp(U.curve, name);
        if ~any(idx), continue; end

        xVals = U.x(idx);
        yVals = U.mean(idx);
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
    title('Phase 2: NMSE versus pilot overhead', 'FontSize', 14);
    legend(hLeg, legLabels, 'Location', 'best', 'FontSize', 10);

    if ~isfolder(outDir), mkdir(outDir); end
    savefig(fig, fullfile(outDir, 'fig_nmse_overhead_matlab.fig'));
    print(fig, fullfile(outDir, 'fig_nmse_overhead_matlab.png'), '-dpng', '-r300');

    fprintf('Saved:\n  %s\n  %s\n', ...
        fullfile(outDir, 'fig_nmse_overhead_matlab.fig'), ...
        fullfile(outDir, 'fig_nmse_overhead_matlab.png'));
end
