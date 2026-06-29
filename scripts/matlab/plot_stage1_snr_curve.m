function plot_stage1_snr_curve()
%PLOT_STAGE1_SNR_CURVE  Plot Stage 1 magnitude NMSE vs SNR.
%
%   Reads the CSV produced by scripts/run_stage1_snr_curve.py, default path:
%   outputs/stage1_snr_curve.csv
%
%   If the CSV is missing, run the Python script first, e.g.:
%       python scripts/run_stage1_snr_curve.py --checkpoint outputs/checkpoints/best.pt

    csvPath = fullfile('..', '..', 'outputs', 'stage1_snr_curve.csv');
    outDir  = fullfile('..', '..', 'outputs');

    if ~isfile(csvPath)
        error(['CSV not found: %s\n' ...
               'Run the Python script first:\n' ...
               '  python scripts/run_stage1_snr_curve.py --checkpoint outputs/checkpoints/best.pt'], ...
              csvPath);
    end

    T = readtable(csvPath);

    lines = {
        'copy_ul',  [0.850 0.325 0.098], '--', 's', 'Copy UL (baseline)';
        'proposed', [0.000 0.447 0.741], '-',  'o', 'Proposed';
    };

    fig = figure('Name', 'Stage 1 Magnitude NMSE vs SNR', 'Color', 'w');
    hold on; box on; grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1);

    hLeg = [];
    legLabels = {};
    for i = 1:size(lines, 1)
        name = lines{i, 1};
        idx = strcmp(T.method, name);
        if ~any(idx), continue; end

        xVals = T.snr_db(idx);
        yVals = T.magnitude_nmse_db(idx);
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
    ylabel('Magnitude NMSE (dB)', 'FontSize', 13);
    title('Stage 1: magnitude NMSE versus SNR', 'FontSize', 14);
    legend(hLeg, legLabels, 'Location', 'best', 'FontSize', 10);

    if ~isfolder(outDir), mkdir(outDir); end
    savefig(fig, fullfile(outDir, 'fig_stage1_snr_curve_matlab.fig'));
    print(fig, fullfile(outDir, 'fig_stage1_snr_curve_matlab.png'), '-dpng', '-r300');

    fprintf('Saved:\n  %s\n  %s\n', ...
        fullfile(outDir, 'fig_stage1_snr_curve_matlab.fig'), ...
        fullfile(outDir, 'fig_stage1_snr_curve_matlab.png'));
end
