data = load("C:/Users/Dresvyanskiy/Desktop/questionnaires/matlabFiles/externalAnnotations_structure.mat")

% Extract the structure containing annotations
annotations_cell = data.external_annotationsAVG;

% Define the annotators
annotators = {'Annotator_1', 'Annotator_2', 'Annotator_3'};

% Loop through each annotator
for annotator_idx = 1:numel(annotators)
    annotator_name = annotators{annotator_idx};

    % Initialize an empty matrix to store the data for the current annotator
    annotator_data = [];

    % Loop through each row in the cell array
    for row_idx = 1:size(annotations_cell, 1)
        disp(['Current row index: ' num2str(row_idx)]);
        % Extract the group number
        group_number = row_idx;

        % Check if the current row has annotations for the current annotator
        if ~isempty(annotations_cell{row_idx, annotator_idx}) % Add 1 to account for the group_number column
            % Extract the annotations for the current annotator
            PLead_values = annotations_cell{row_idx, annotator_idx}.PLead;
            PDom_values = annotations_cell{row_idx, annotator_idx}.PDom;
            % Check if PLead_values is 1x3, if so, append NaN
            if numel(PLead_values) == 3
                PLead_values = [PLead_values NaN];
            end

            % Check if PDom_values is 1x3, if so, append NaN
            if numel(PDom_values) == 3
                PDom_values = [PDom_values NaN];
            end
        else
            % If there are no annotations, fill with NaNs
            PLead_values = NaN(1, 4);
            PDom_values = NaN(1, 4);
        end

        % Concatenate group number and annotations
        row_data = [group_number, PLead_values, PDom_values];

        % Append the row data to the annotator_data matrix
        annotator_data = [annotator_data; row_data];
    end

    % Create headers
    headers = {'group_number', 'PLead_1', 'PLead_2', 'PLead_3', 'PLead_4', 'PDom_1', 'PDom_2', 'PDom_3', 'PDom_4'};

    % Save the data for the current annotator as a CSV file
    csvwrite_with_headers(['Annotator_' num2str(annotator_idx) '_annotations.csv'], annotator_data, headers);
end

