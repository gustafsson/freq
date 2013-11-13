function settings=exportpeakfrequencies_settings()
  settings.chunk_size = -1; % entire signal in one pass
  settings.arguments = '129, ''peaks.csv''';
  settings.argument_description = 'Number of frequencies and filename. Example: "129, ''peaks.csv''"';
  settings.icon = 'Export frequencies';
end
