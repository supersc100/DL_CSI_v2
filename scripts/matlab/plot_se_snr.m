function plot_se_snr()
%PLOT_SE_SNR  Plot spectral efficiency vs SNR from run_simulation.py CSV.
%
%   Reads outputs/sim/results_se_snr.csv.

    csvPath = fullfile('..', '..', 'outputs', 'sim', 'results_se_snr.csv');
    outDir  = fullfile('..', '..', 'outputs', 'sim');

    if ~isfile(csvPath)
        error('CSV not found: %s', csvPath);
    end

    T = readtable(csvPath);

    lines = {
        'proposed',    [0.000 0.447 0.741], '-',  'o', 'Proposed';
        'perfect_csi', [0.850 0.325 0.098], '--', 's', 'Perfect CSI (upper bound)';
    };

    fig = figure('Name', 'Spectral Efficiency vs SNR', 'Color', 'w');
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

    xlabel('SNR (dB)', 'FontSize', 13);
    ylabel('Spectral efficiency (bits/s/Hz)', 'FontSize', 13);
    title('Spectral efficiency versus SNR', 'FontSize', 14);
    legend(hLeg, legLabels, 'Location', 'best', 'FontSize', 10);

    if ~isfolder(outDir), mkdir(outDir); end
    savefig(fig, fullfile(outDir, 'fig_se_snr_matlab.fig'));
    print(fig, fullfile(outDir, 'fig_se_snr_matlab.png'), '-dpng', '-r300');

    fprintf('Saved:\n  %s\n  %s\n', ...
        fullfile(outDir, 'fig_se_snr_matlab.fig'), ...
        fullfile(outDir, 'fig_se_snr_matlab.png'));
end
