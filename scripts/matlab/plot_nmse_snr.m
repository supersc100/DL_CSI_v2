function plot_nmse_snr()
%PLOT_NMSE_SNR  Plot Phase 2 NMSE vs SNR from run_simulation.py CSV.
%
%   Reads outputs/sim/results_nmse_snr.csv and produces an editable MATLAB
%   figure plus a PNG.  Line styles, colors and markers are easy to change
%   in the LINES table below.
%
%   Run in VS Code: open this file and press the MATLAB "Run" button, or
%   type in the terminal:
%       matlab -batch "plot_nmse_snr"

    csvPath = fullfile('..', '..', 'outputs', 'sim', 'results_nmse_snr.csv');
    outDir  = fullfile('..', '..', 'outputs', 'sim');

    if ~isfile(csvPath)
        error('CSV not found: %s', csvPath);
    end

    T = readtable(csvPath);

    % Column description:
    %   curve  -> method name
    %   x      -> SNR in dB
    %   metric -> 'nmse_db'
    %   mean   -> mean NMSE in dB
    %   std    -> standard deviation across seeds

    snr = T.x;
    nmse = T.mean;
    curves = unique(T.curve, 'stable');

    % --- styling lookup ---------------------------------------------------
    % Each row: {curve_name, Color, LineStyle, Marker, DisplayName}
    lines = {
        'proposed',         [0.000 0.447 0.741], '-',  'o',  'Proposed';
        'proposed_quant_16bit', [0.301 0.745 0.933], '--', 's',  'Proposed (16-bit Q)';
        'proposed_quant_8bit',  [0.301 0.745 0.933], '-.', '^',  'Proposed (8-bit Q)';
        'proposed_quant_4bit',  [0.301 0.745 0.933], ':',  'd',  'Proposed (4-bit Q)';
        'proposed_quant_2bit',  [0.635 0.078 0.184], '--', 'v',  'Proposed (2-bit Q)';
        'magnitude_only',   [0.466 0.674 0.188], '-',  'x',  'Magnitude only';
        'linear_interp',    [0.494 0.184 0.556], '--', '>',  'Linear interp';
        'dft_interp',       [0.850 0.325 0.098], '-.', '<',  'DFT interp';
        'full_feedback',    [0.500 0.500 0.500], ':',  'none', 'Full feedback (upper bound)';
    };

    fig = figure('Name', 'Phase 2 NMSE vs SNR', 'Color', 'w');
    hold on; box on; grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1);

    hLeg = [];
    legLabels = {};
    for i = 1:size(lines, 1)
        name = lines{i, 1};
        idx = strcmp(T.curve, name);
        if ~any(idx), continue; end

        [xSort, order] = sort(snr(idx));
        ySort = nmse(idx);
        ySort = ySort(order);

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
    ylabel('NMSE (dB)', 'FontSize', 13);
    title('Phase 2: Full-CSI NMSE versus SNR', 'FontSize', 14);
    legend(hLeg, legLabels, 'Location', 'best', 'FontSize', 10);
    xlim([min(snr)-2, max(snr)+2]);

    % Make output directory if necessary
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    % Save editable figure and high-resolution PNG
    savefig(fig, fullfile(outDir, 'fig_nmse_snr_matlab.fig'));
    print(fig, fullfile(outDir, 'fig_nmse_snr_matlab.png'), '-dpng', '-r300');

    fprintf('Saved:\n  %s\n  %s\n', ...
        fullfile(outDir, 'fig_nmse_snr_matlab.fig'), ...
        fullfile(outDir, 'fig_nmse_snr_matlab.png'));
end
