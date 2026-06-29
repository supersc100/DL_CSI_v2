%RUN_ALL_PLOTS  Regenerate all MATLAB figures from existing simulation CSVs.
%
%   Run in VS Code:
%   - Open this file and click the MATLAB "Run" button, OR
%   - In the terminal:  matlab -batch "run('scripts/matlab/run_all_plots.m')"

clc; close all;

fprintf('=== Regenerating all MATLAB figures ===\n\n');

plot_nmse_snr();
plot_nmse_overhead();
plot_sampling_overhead();
plot_se_snr();

% Stage 1 curve requires the CSV from run_stage1_snr_curve.py.
stage1Csv = fullfile('..', '..', 'outputs', 'stage1_snr_curve.csv');
if isfile(stage1Csv)
    plot_stage1_snr_curve();
else
    fprintf(['\nSkipping Stage 1 SNR curve: %s not found.\n' ...
             'Generate it with:\n' ...
             '  python scripts/run_stage1_snr_curve.py --checkpoint outputs/checkpoints/best.pt\n'], ...
             stage1Csv);
end

fprintf('\n=== Done ===\n');
