#require("Date")

#$platform = "macos_i386"
#$build_date = Date.today
#$version = "0.%d.%02d.%02d" % [$build_date.year, $build_date.month, $build_date.mday]
$version = ARGV[0]
$build_name = "sonicawe_#{$version}_snapshot_#{$platform}"

$dist_dir = "mac_dist"
$build_dir = "../../sandbox"

# Creating a dist directory
system("mkdir #{$dist_dir}") if( !File.exist?($dist_dir) )

# Copy the build and rename it
$app_name = "#{$dist_dir}/#{$build_name}.app"
system("rm -r #{$app_name}") if( File.exist?($app_name) )
puts "Copying build to #{$app_name}"
system("cp -r #{$build_dir} #{$app_name}") if( File.exist?($build_dir) )

# Set the QT library search paths
$unix_app_name = "#{$app_name}/Contents/MacOS/sonicawe"
puts "Setting QT library search paths"
system("install_name_tool -change QtOpenGL.framework/Versions/4/QtOpenGL @executable_path/../Frameworks/QtOpenGL #{$unix_app_name}\n
install_name_tool -change QtGui.framework/Versions/4/QtGui @executable_path/../Frameworks/QtGui #{$unix_app_name}\n
install_name_tool -change QtCore.framework/Versions/4/QtCore @executable_path/../Frameworks/QtCore #{$unix_app_name}\n
install_name_tool -change @rpath/libcufft.dylib @executable_path/../Frameworks/libcufft.dylib #{$unix_app_name}\n
install_name_tool -change @rpath/libcudart.dylib @executable_path/../Frameworks/libcudart.dylib #{$unix_app_name}")

# Update version string in Info.plist
puts "Updating the Info.plist with version number #{$version}"
$info_name = "#{$app_name}/Contents/Info.plist"
$info = File.read($info_name)
$info.gsub!("(VERSION_TAG)", "#{$version} (Snapshot)")
$info.gsub!("(LONG_VERSION_TAG)", "SonicAwe #{$version} (Snapshot)")
File.open($info_name, "w") do |file|
    file.write($info)
end

# Packaging the application (using zip)
$packet_name = "#{$dist_dir}/#{$build_name}.zip"
system("rm -r #{$packet_name}") if( File.exist?($packet_name) )
puts "Packaging application: #{$packet_name}"
system("zip -r #{$packet_name}  #{$app_name}")